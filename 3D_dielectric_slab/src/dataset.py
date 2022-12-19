import numpy as np
import pandas as pd

from mindelec.data import Dataset, ExistedDataConfig
from mindelec.geometry import Cuboid
from mindelec.geometry import create_config_from_edict

from src.config import maxwell_3d_config, cuboid_sampling_config
# from config import maxwell_3d_config, cuboid_sampling_config

def preprocessing_waveguide_data(waveguide_data_path, npy_save_path,
                                label_save_path):
    """预处理Modulus的csv数据，使其转换成npy格式"""
    df = pd.read_csv(waveguide_data_path)
    print(df.head())
    # 该csv文件共8列，前两列为(x,y)为x,y坐标，后六列(u0,u1,...,u5)为独立的测量值
    # 根据Modulus例子，取u0列作为波导面的垂直方向分布
    df["z"] = -0.5 * np.ones(len(df))
    df_need = df[["z", "x", "y"]]
    print(df_need.head())
    points = df_need.to_numpy(dtype=np.float32)
    label = df[[f"u{i}" for i in range(6)]].to_numpy(dtype=np.float32)
    
    print(points[:5, :])
    print(label[:5,:])
    np.save(npy_save_path, points)
    np.save(label_save_path, label)
    

def create_train_dataset()->Dataset:
    """create trainning dataset from existed data and sample"""
    # 左侧波导管数据
    npy_points_path = maxwell_3d_config["waveguide_points_path"]
    waveguide_port = ExistedDataConfig(name=maxwell_3d_config["waveguide_name"],
                                      data_dir=[npy_points_path],
                                      columns_list=["points"],
                                      data_format="npy",
                                      constraint_type="Label",
                                      random_merge=False)
    # 其他位置数据
    cuboid_space = Cuboid(name=maxwell_3d_config["geom_name"],
                        coord_min=maxwell_3d_config["coord_min"],
                        coord_max=maxwell_3d_config["coord_max"],
                        sampling_config=create_config_from_edict(cuboid_sampling_config))
    geom_dict = {cuboid_space: ["domain", "BC"]}

    # create dataset for train and test
    train_dataset = Dataset(geom_dict, existed_data_list=[waveguide_port])
    # train_dataset = Dataset(geom_dict)

    return train_dataset


if __name__ == "__main__":
    waveguide_data_path = "../../validation/2Dwaveguideport.csv"
    npy_save_path = "../../validation/sample_points.npy"
    label_save_path = "../../validation/sample_points_label.npy"
    preprocessing_waveguide_data(waveguide_data_path, npy_save_path,label_save_path)
    
    """
    train_dataset = create_train_dataset(npy_save_path, label_save_path)
    train_data = train_dataset.create_dataset(batch_size=128)
    
    print("dataset_columns_map", train_dataset.dataset_columns_map)
    print("column_index_map", train_dataset.column_index_map)
    print("dataset_constraint_map", train_dataset.dataset_constraint_map)
    
    for ds in train_dataset.all_datasets:
        print(ds.name)
        print(ds.constraint_type)
        print(ds.columns_list,'\n')
    
        
    print(train_dataset.get_columns_list())
    """