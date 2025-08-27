import sys
import numpy as np
import matplotlib.pyplot as plt
from syntheisze_cbfs import plot_cbf_level_sets

def load_data(file_path):
    return np.load(file_path, allow_pickle=True).item()

def analyze():
    file_path = sys.argv[1]

    x_lim = [-1.2, 1.2]
    y_lim = [-1.5, 1.5]

    with open ('configs/config.json', 'r') as f:
        config = json.load(f)

    P = np.diag(config['P_diag'])
    state_dim = P.shape[0]
    c = np.array(config['c']).reshape((state_dim, 1))
    
    data = load_data(file_path)
    keys = list(data.keys())
    print(f"len(keys): {len(keys)}")
    print(f"max(keys): {max(keys)}")
    
    # extract the bbox_volume from the data
    bbox_volumes = [data[key]['bbox_volume'] for key in data]
    # print(bbox_volumes)
    # plot the bbox_volumes
    plt.hist(bbox_volumes, bins=100)
    plt.xlabel("Bbox Volume")
    plt.ylabel("Frequency")
    plt.title("Bbox Volume Distribution")
    plt.grid(True)
    path_parts = file_path.split("/")
    file_name = path_parts[-1].split(".")[0]
    plt.savefig(f"{'/'.join(path_parts[:-1])}/bbox_volumes_{file_name}.png")
    plt.close()

    # extract the largest bbox_volume
    sorted_indices = np.argsort(bbox_volumes)[::-1]
    largest_bbox_volume_idx = int(sorted_indices[0])  # Get largest
    largest_bbox_volume = bbox_volumes[largest_bbox_volume_idx]
    print(f"largest_bbox_volume: {largest_bbox_volume}")
    # extract the P_list and c_list of the largest bbox_volume
    P_list = data[keys[largest_bbox_volume_idx]]['P_list']
    c_list = data[keys[largest_bbox_volume_idx]]['c_list']
    c_list = [c.reshape(-1, 1) for c in c_list]
    print(f"P_list: {P_list}")
    print(f"c_list: {c_list}")

    # plot the P_list and c_list
    plot_cbf_level_sets(P, c, P_list, c_list, [x_lim, y_lim])

    # Save the best P_list and c_list
    np.save(f"{'/'.join(path_parts[:-1])}/best_P_list_{file_name}.npy", P_list)
    np.save(f"{'/'.join(path_parts[:-1])}/best_c_list_{file_name}.npy", c_list)

if __name__ == "__main__":
    analyze()