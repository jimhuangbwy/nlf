import asyncio
import tkinter as tk
from winrt.windows.services.store import StoreContext
from winrt.runtime.interop import initialize_with_window


# --- The Async Purchase Logic ---
async def buy_product_logic(root, store_id):
    print("Initializing Store Context...")
    context = StoreContext.get_default()

    # 1. Get the HWND (Window Handle) from Tkinter
    # This tells the Store dialog to attach to our hidden Python window
    hwnd = int(root.frame(), 16)

    # 2. Initialize with Window
    try:
        initialize_with_window(context, hwnd)
    except Exception as e:
        print(f"Failed to initialize window: {e}")
        return

    print(f"Requesting purchase for {store_id}...")
    print("PLEASE NOTE: A Windows Store dialog should appear now.")

    # 3. Request Purchase
    # We await this, but our custom loop below ensures the UI keeps updating
    result = await context.request_purchase_async(store_id)

    # 4. Handle Results
    if result.status == 1:  # Succeeded
        print("\n*** PURCHASE SUCCESSFUL ***")
    elif result.status == 2:  # AlreadyPurchased
        print("\n*** ALREADY OWNED ***")
    elif result.status == 3:  # NotPurchased
        print("\n*** CANCELED BY USER ***")
    else:
        print(f"\n*** FAILED: Status {result.status} ***")
        if result.extended_error:
            print(f"Error: {result.extended_error}")


# --- The Custom UI Loop (Fixes the Error) ---
def run_ui_loop(coroutine, root):
    """
    This runs the asyncio task AND pumps Tkinter window messages
    at the same time.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    task = loop.create_task(coroutine)

    try:
        # Loop until the async task is finished
        while not task.done():
            # 1. Run a tiny bit of the async loop
            loop.run_until_complete(asyncio.sleep(0.1))

            # 2. Pump Windows Messages (Critical for the Store Dialog to work)
            root.update()

    except Exception as e:
        print(f"Loop error: {e}")
    finally:
        # Clean up
        root.destroy()
        # Get the result (or raise the exception if it failed)
        if not task.cancelled() and task.exception():
            raise task.exception()


if __name__ == "__main__":
    MY_PRODUCT_ID = "9PLVVPGD3FV7"  # Example ID (or use your own)

    # 1. Create a hidden Tkinter window to serve as the "Parent"
    root = tk.Tk()
    root.withdraw()  # Hide the main tiny window, we only need it for the HWND

    # 2. Run our custom loop
    try:
        run_ui_loop(buy_product_logic(root, MY_PRODUCT_ID), root)
    except Exception as e:
        print(f"Application Error: {e}")