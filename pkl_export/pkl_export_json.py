import pickle
import json

f = open('../nlf_data_files/joint_info_866.pkl', 'rb')
joint_info = pickle.load(f)

data = joint_info.n_joints

print(type(data), data)

# data = data.to_dict()
#
# with open("joint_info_ids.json", "w") as f:
#     json.dump(data, f, indent=2)