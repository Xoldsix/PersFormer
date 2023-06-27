"""
    划分数据集
"""

import os
import cv2
import json
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def save_img(oss, save_dir):
    clipid = oss[-87:-32]
    img_path = os.path.join(save_dir, clipid + ".jpg")
    with open(oss, "r") as fin:
        j4d = json.load(fin)
        for camera in j4d["camera"]:
            if camera.get("name", "") in ["front_middle_camera", "front_wide_camera"]:
                img = cv2.imread(os.path.join("/", camera["oss_path"]))
                # h, w = img.shape[:2]
                # out_w = int(w * 648 / h)
                # img = cv2.resize(img, (out_w, 648))
                cv2.imwrite(img_path, img)  
                files = img_path.split("/")[-3:]
                return os.path.join(files[0], files[1], files[2]), img
            
def load_lanes(oss, lidar2car):
    with open(oss, "r") as fin:
        j4d = json.load(fin)
        car_lanes, lanes, gt_lanes = [], [], []
        for obj in j4d["labeled_data"]["objects"]:
            if obj["class_name"] != "lane":
                continue
            lane = []
            for child in obj["children"]:
                points = [[p["x"], p["y"], p["z"]] for p in child["geometry"]["points"]]
                points = np.array(points).astype("float32")
                points = interpolate(points)
                car_points = convert_coordinate(points, lidar2car=lidar2car).tolist()
                for x,y,z in car_points:
                    lane.append([-y, x, z])
                globe_points = points.tolist()
            car_lanes.append(car_points)
            lanes.append(globe_points)
            gt_lanes.append(lane)

    return car_lanes, lanes, gt_lanes

def interpolate(points, delta=0.5):
        npoints = []
        for p0, p1 in zip(points[0: -1], points[1: ]):
            dist = math.sqrt(np.sum([x ** 2 for x in p1 - p0]))
            if dist < 2 * delta:
                npoints.append(p0)
                continue

            for idx in range(int(dist / delta)):
                npoints.append(p0 + idx * (p1 - p0) / int(dist / delta))

        if npoints[-1] is not points[-1]:
            npoints.append(points[-1])

        return np.array(npoints)

def convert_coordinate(points, lidar2car):
    return np.dot(points, lidar2car[: 3, : 3].T) + lidar2car[: 3, 3]

def Rt_T(R, t):
    T = np.eye(4, dtype="float32")
    T[0: 3, 0: 3] = R[:, :]
    T[0: 3, 3] = t[:]

    return T

def load_cam_pas(oss):
    with open(oss, "r", encoding="utf-8") as f:
        j4d = json.loads(f.read())
        hardware_config = os.path.join("/", j4d["hardware_config_path"])
    with open(hardware_config, "r", encoding="utf-8") as fin:
        jconfig = json.load(fin, strict=False)
        exlidar = parse_camera_external_param(jconfig["sensor_config"]["lidar_param"])
        exrtk = parse_camera_external_param(jconfig["sensor_config"]["rtk_param"])
        excam = parse_camera_external_param(jconfig["sensor_config"]["cam_param"])
        incam = parse_camera_internal_param(jconfig["sensor_config"]["cam_param"])

        R_lidar2car = exlidar["MIDDLE_LIDAR"]["R"]
        t_lidar2car = exlidar["MIDDLE_LIDAR"]["T"]
        lidar2car = Rt_T(R_lidar2car, t_lidar2car)
        
        R_cam2car = excam["front_middle_camera"]["R"]
        t_cam2car = excam["front_middle_camera"]["T"]
        T_cam2car = Rt_T(R_cam2car, t_cam2car)

        K = np.array([[incam["front_middle_camera"]["fx"], 0, incam["front_middle_camera"]["cx"]],
                    [0, incam["front_middle_camera"]["fy"], incam["front_middle_camera"]["cy"]],
                    [0, 0, 1]])
        imsize = [incam["front_middle_camera"]["image_width"], incam["front_middle_camera"]["image_height"]]


        cam_height = excam["front_middle_camera"]["cam_height"]
        cam_pitch = excam["front_middle_camera"]["pitch"]

        return cam_height, cam_pitch, lidar2car, K, T_cam2car, imsize, R_cam2car.tolist()

def parse_camera_external_param(params):
    ret = {}
    for param in params:
        if "name" not in param or "pose" not in param:
            continue

        quat = [param["pose"]["attitude"].get(axis, 0.0) for axis in "xyzw"]
        rotation = R.from_quat(quat).as_matrix()

        trans = [param["pose"]["translation"].get(axis, 0.0) for axis in "xyz"]
        ypr = {axis: param["pose"]["attitude_ypr"].get(axis, 0.0) for axis in ["pitch", "yaw", "roll"]}

        ret[param["name"]] = {"R": rotation, "T": np.array(trans), "cam_height": param["pose"]["translation"]["z"]}
        ret[param["name"]].update(ypr)

    return ret

def parse_camera_internal_param(params):
        ret = {}
        for param in params:
            if "name" not in param:
                continue

            infos = {k: param[k] for k  in ["cx", "cy", "distortion", "fx", "fy", "image_height", "image_width"]}

            ret[param["name"]] = infos

        return ret

def write_json(name, data_dict):
    with open(name, "w", encoding='utf-8') as f:
        for item in data_dict:
            f.write(json.dumps(item) + "\n")

def load_dict(oss, img_path):
    data_dict, label_dict = {}, {}
    
    cam_height, cam_pitch, lidar2car, K, T_cam2car, imsize, R_cam2car = load_cam_pas(oss)
    car_lanes, lanes, gt_lanes = load_lanes(oss, lidar2car)
    for i in range(len(gt_lanes)-1, -1, -1):
        gt_lanes[i] = np.array(gt_lanes[i])
        valid_mask = (gt_lanes[i][:, 0] >= -20) & (gt_lanes[i][:, 0] < 20) \
                    & (gt_lanes[i][:, 1] >= 0) & (gt_lanes[i][:, 1] < 200)

        (row_idxs, *_) = np.where(valid_mask)
        if len(gt_lanes[i][row_idxs, :]) <= 1:
            gt_lanes.pop(i)
            continue
        gt_lanes[i] = gt_lanes[i][row_idxs, :].tolist()
    K = K.tolist()
    T_cam2car = T_cam2car.tolist()

    laneLines_visibility = []
    for lane in gt_lanes:
        visibility = []
        for point in lane:
            visibility.append(1.0)
        laneLines_visibility.append(visibility)

    # data dict
    data_dict["raw_file"] = img_path
    data_dict["cam_height"] = cam_height
    data_dict["cam_pitch"] = cam_pitch
    data_dict["laneLines"] = gt_lanes
    data_dict["laneLines_visibility"] = laneLines_visibility
    data_dict["centerLines"] = gt_lanes
    data_dict["K"] = K
    data_dict["imsize"] = imsize
    data_dict["T_cam2car"] = T_cam2car
    data_dict["R"] = R_cam2car
    data_dict["car_points"] = car_lanes
    
    # label dict
    label_dict["cam_pitch"] = cam_pitch
    label_dict["cam_height"] = cam_height
    label_dict["intrinsic"] = K
    label_dict["extrinsic"] = T_cam2car
    label_dict["K"] = K
    label_dict["T_cam2car"] = T_cam2car
    label_dict["R"] = R_cam2car

    return data_dict, json.dumps(label_dict)

def save_label(save_dir,label_dict, oss):
    clip = oss[-87:-32]
    table_path = os.path.join(save_dir, clip + ".txt") 
    with open(table_path, "w") as f:
        f.write(label_dict)

def main(data_path, save_dir, num=36):
    img_save_path = os.path.join(save_dir, "images", "train")
    labels_save_path = os.path.join(save_dir, "labels", "train")
    os.makedirs(img_save_path)
    os.makedirs(labels_save_path)
    train_name = os.path.join(save_dir, "train.json")
    test_name = os.path.join(save_dir, "test.json")
    with open(data_path, "r",  encoding='utf-8') as fd:
        nn, dict_all = 0, []
        for line in fd.readlines():
            if nn < num:
                oss = line.strip()
                img_path, img = save_img(oss, img_save_path)
                data_dict, label_dict = load_dict(oss, img_path)
                save_label(labels_save_path, label_dict, oss)
                dict_all.append(data_dict)
                nn += 1
                # get_mask(data_dict["laneLines"], oss)
                # draw_lane(img, data_dict, oss)

            elif  nn < 2*num:
                oss = line.strip()
                img_path, img = save_img(oss, img_save_path)
                data_dict, label_dict = load_dict(oss, img_path)
                save_label(labels_save_path, label_dict, oss)
                dict_all.append(data_dict)
                nn += 1

            if nn == num:
                write_json(train_name, dict_all)
                img_save_path = os.path.join(save_dir, "images", "test")
                labels_save_path = os.path.join(save_dir, "labels", "test")
                os.makedirs(img_save_path)
                os.makedirs(labels_save_path)
                dict_all = []  
            if nn == 2*num:
                write_json(test_name, dict_all)
                break

def draw_lane(img, data_dict, oss):
    clip = oss[-87:-32]
    img_mask=np.zeros((515, 240, 3),np.uint8)
    lanes = data_dict["laneLines"]
    for lane in lanes:
        for i in range(len(lane) - 1):
            x, y, z = lane[i]
            # x = 515 - (x*5 + 115)
            # y = 240 - (y*10 + 120)
            x2, y2, z2 = lane[i + 1]
            # x2 = 515 - (x2*5 + 115)
            # y2 = 240 - (y2*10 + 120)
            cv2.circle(img_mask, (int(y), int(x)), 1, (255,255,255), -1)
            cv2.circle(img_mask, (int(y2), int(x2)), 1, (255,255,255), -1)
            cv2.line(img_mask, (int(y), int(x)), (int(y2), int(x2)) , (0,0,0), 1)
    cv2.circle(img_mask, (120, 0), 1, (0,0,255), -1)
    cv2.circle(img_mask, (120, 515), 1, (0,0,255), -1)   
    cv2.line(img_mask, (120, 0), (120, 515) , (0,0,255), 1)
    cv2.line(img_mask, (0, 400), (240, 400) , (0,0,255), 1)

    hm, wm = img_mask.shape[0], img_mask.shape[1] # 515, 240
    h, w = img.shape[:2]
    out_w = int(w * hm / h) 
    imgH1 = cv2.resize(img, (out_w, hm))
    imgStackH = cv2.hconcat((imgH1, img_mask))
    # print(save_img_path)
    cv2.imwrite(f"/mnt/ve_perception/likaiying/opendataset/sandbox/PersFormer_3DLane_main/haomodata/%s.jpg"%(clip), imgStackH)

def get_mask(center_points_list, oss):
    clip = oss[-87:-32]
    plt.figure(figsize=(6.0, 12.0))
    plt.xlim(-30, 30)
    plt.ylim(0, 120)
    # plt.yticks(range(-20,101,5))
    plt.title('gt map')

    plt.gca().set_aspect(1)
    if center_points_list is not None:
        for center_points in center_points_list:
            center_points = np.array(center_points)
            xs = center_points[:, 0]
            ys = center_points[:, 1]
            plt.plot(xs, ys, color="r")
    xs = np.array([-10, -10, 0, 0])
    ys = np.array([80, 60, 60, 80])
    plt.plot(xs, ys, color="green")
    plt.grid(visible=True)
    plt.savefig(f"/mnt/ve_perception/likaiying/opendataset/sandbox/PersFormer_3DLane_main/haomodata/%s.jpg"%(clip))
    plt.close()
            


if __name__ == '__main__':
    save_dir = "/mnt/ve_perception/likaiying/opendataset/sandbox/haomodata"
    data_path = "/mnt/ve_perception/likaiying/opendataset/sandbox/lanedata_6v_30/curve/curve/curve.txt"
    main(data_path, save_dir, num=18)
