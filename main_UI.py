import sys
import asyncio
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QComboBox, QScrollArea, QFrame, QMessageBox)
from PySide6.QtCore import Qt, QSize, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QColor, QPalette

# WinRT Imports
from winrt.windows.services.store import StoreContext
from winrt.runtime.interop import initialize_with_window

# Async Bridge for Qt
from qasync import QEventLoop, asyncSlot

# --- IMPORT TRACKING LOGIC ---
try:
    from pino_tracker import PinoTracker
except ImportError:
    print("Could not import main.py. Ensure it is in the same directory.")
    PinoTracker = None  # Fallback to prevent crash if file missing


# --- Worker Thread for Tracking ---
# We use a QThread to handle the "Start" command because loading models takes time
# and we don't want to freeze the GUI.
class TrackingWorker(QThread):
    finished = Signal()

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def run(self):
        # This runs in a background thread
        # It loads models (if not loaded) and starts the CV loop
        self.tracker.start()
        self.finished.emit()


# --- Custom Widget for a Single Product Item ---
class ProductItem(QFrame):
    def __init__(self, title, description, price, product_id, parent_callback):
        super().__init__()
        self.product_id = product_id
        self.callback = parent_callback

        self.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 1px solid #3e3e3e;
                border-radius: 5px;
            }
            QFrame:hover {
                background-color: #3a3a3a;
                border: 1px solid #ffee00;
            }
            QLabel {
                border: none;
                background: transparent;
            }
        """)
        self.setFixedHeight(80)

        layout = QHBoxLayout(self)

        text_layout = QVBoxLayout()
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("color: #ffee00; font-size: 16px; font-weight: bold;")
        self.lbl_desc = QLabel(description)
        self.lbl_desc.setStyleSheet("color: #cccccc; font-size: 12px;")
        text_layout.addWidget(self.lbl_title)
        text_layout.addWidget(self.lbl_desc)

        self.lbl_price = QLabel(price)
        self.lbl_price.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        self.lbl_price.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addLayout(text_layout)
        layout.addWidget(self.lbl_price)

    def mousePressEvent(self, event):
        self.callback(self.product_id, self.lbl_title.text())


# --- The Purchase Page UI ---
class PurchasePage(QWidget):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(10, 10, 10, 10)

        lbl_header = QLabel("UPGRADE ITEMS")
        lbl_header.setStyleSheet("color: grey; font-size: 14px; margin-top: 10px;")
        self.layout.addWidget(lbl_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: #121212;")
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self.scroll_content)
        self.layout.addWidget(scroll)

        lbl_warning = QLabel("Microsoft store does not support refund. Be cautious of any purchases.")
        lbl_warning.setStyleSheet("color: red; font-size: 14px; font-weight: bold; margin-top: 10px;")
        self.layout.addWidget(lbl_warning)

        self.target_product_ids = ["9PLVVPGD3FV7", "9MXX7VCG599M"]
        QTimer.singleShot(0, self.load_store_products)

    @asyncSlot()
    async def load_store_products(self):
        loading_lbl = QLabel("Connecting to Microsoft Store...")
        loading_lbl.setStyleSheet("color: white; font-style: italic;")
        self.scroll_layout.addWidget(loading_lbl)

        try:
            context = StoreContext.get_default()
            hwnd = int(self.parent_window.winId())
            initialize_with_window(context, hwnd)
            product_kinds = ["Durable", "Consumable"]
            result = await context.get_store_products_async(product_kinds, self.target_product_ids)

            if result.extended_error and result.extended_error.value != 0:
                print(f"Store API Error: {result.extended_error}")
            elif result.products and len(result.products) > 0:
                print(f"-> Success! Found {len(result.products)} products.")
                loading_lbl.deleteLater()
                for store_id, product in result.products.items():
                    title = product.title
                    price = product.price.formatted_recurrence_price
                    desc = product.description if product.description else "Unlock knees and ankles"
                    item = ProductItem(title, desc, price, store_id, self.initiate_purchase)
                    self.scroll_layout.addWidget(item)
            else:
                print("-> Success (Connected), but no matching products found.")
                loading_lbl.setText("No active subscriptions found in this region.")

        except Exception as e:
            print(f"Crash during fetch: {e}")
            loading_lbl.setText("Store Connection Failed.")

    @asyncSlot()
    async def initiate_purchase(self, store_id, product_name):
        try:
            context = StoreContext.get_default()
            hwnd = int(self.parent_window.winId())
            initialize_with_window(context, hwnd)
            result = await context.request_purchase_async(store_id)
            if result.status == 1:
                QMessageBox.information(self, "Success", f"Purchased: {product_name}")
            elif result.status == 2:
                QMessageBox.warning(self, "Owned", "You already own this item.")
            elif result.status == 3:
                print("User cancelled.")
            else:
                QMessageBox.critical(self, "Error", f"Failed with status: {result.status}")
        except Exception as e:
            QMessageBox.critical(self, "System Error", str(e))


# --- Main Window with Tab Bar ---
class ModernTabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PinoFBT 2.0")
        self.resize(600, 500)
        self.setStyleSheet("QMainWindow { background-color: #000000; }")

        # Initialize Tracking Logic
        self.tracker = None
        self.is_tracking = False
        if PinoTracker:
            self.tracker = PinoTracker()

        # Tab Setup
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South)
        self.setCentralWidget(self.tabs)

        # Create Pages
        self.create_home_page()
        self.purchase_page = PurchasePage(self)
        self.purchase_page.setStyleSheet("background-color: black;")

        self.tabs.addTab(self.home_page, "Home")
        self.tabs.addTab(self.purchase_page, "Subscription")

        self.apply_tab_styles()

    def create_home_page(self):
        self.home_page = QWidget()
        self.home_page.setStyleSheet("background-color: black;")

        layout = QVBoxLayout(self.home_page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # LINK BUTTON
        self.btn_link = QPushButton("START TRACKING")
        self.btn_link.setFixedSize(250, 60)
        self.btn_link.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.btn_link.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_link.setStyleSheet("""
            QPushButton {
                background-color: #ffee00;
                color: black;
                border-radius: 30px;
                border: none;
            }
            QPushButton:hover {
                background-color: #ffdd00;
            }
            QPushButton:pressed {
                background-color: #e6c300;
            }
        """)
        self.btn_link.clicked.connect(self.toggle_tracking)

        # Status Label
        self.lbl_status = QLabel("Ready to connect")
        self.lbl_status.setStyleSheet("color: gray; margin-top: 15px; font-size: 14px;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.btn_link)
        layout.addWidget(self.lbl_status)

    def toggle_tracking(self):
        if not self.tracker:
            QMessageBox.critical(self, "Error", "Tracking module not loaded correctly.")
            return

        if not self.is_tracking:
            # START
            self.lbl_status.setText("Loading Models... (This may take a moment)")
            self.btn_link.setEnabled(False)  # Prevent double click

            # Run in background thread to keep UI alive
            self.worker = TrackingWorker(self.tracker)
            self.worker.finished.connect(self.on_tracking_started)
            self.worker.start()
        else:
            # STOP
            self.tracker.stop()
            self.is_tracking = False
            self.btn_link.setText("START TRACKING")
            self.btn_link.setStyleSheet("""
                QPushButton { background-color: #ffee00; color: black; border-radius: 30px; }
                QPushButton:hover { background-color: #ffdd00; }
            """)
            self.lbl_status.setText("Tracking Stopped")

    def on_tracking_started(self):
        self.is_tracking = True
        self.btn_link.setEnabled(True)
        self.btn_link.setText("STOP TRACKING")
        self.btn_link.setStyleSheet("""
            QPushButton { background-color: #cc0000; color: white; border-radius: 30px; }
            QPushButton:hover { background-color: #aa0000; }
        """)
        self.lbl_status.setText("Tracking Active (Check Popup Window)")

    def apply_tab_styles(self):
        self.setStyleSheet("""
            QTabWidget::pane { border: 0px; background: #000000; }
            QTabBar::tab {
                background: #000000;
                color: white;
                padding: 15px;
                min-width: 150px;
                font-size: 14px;
                border-top: 2px solid #333;
            }
            QTabBar::tab:selected {
                color: #ffee00;
                border-top: 2px solid #ffee00;
            }
            QTabBar::tab:hover { background: #1a1a1a; }
        """)

    def closeEvent(self, event):
        # Ensure tracking threads stop when closing the app
        if self.tracker and self.is_tracking:
            self.tracker.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = ModernTabWindow()
    window.show()
    with loop:
        loop.run_forever()