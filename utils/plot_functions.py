import matplotlib.pyplot as plt
import numpy as np
import random
import math
import io
from os import path
from Structure.Building import Building
from Structure.Scene import Scene
from PIL import Image

SHAPE = (1024, 1024)
my_dpi = 128


def plot_sample_img(scene, image_outpath, image_count):
    noise = np.random.normal(255. / 2, 255. / 10, SHAPE)

    img = plt.figure(figsize=(1330 / my_dpi, 1330 / my_dpi), dpi=my_dpi, frameon=False)
    axes = plt.gca()
    axes.imshow(noise, extent=[-1024, 1024, -1024, 1024])
    axes.set_xlim([0, 1024])
    axes.set_ylim([0, 1024])
    axes.axis('off')

    for structure in scene.buildings:
        img_matrix = structure.vertices
        center = structure.center
        a, b, c, d = img_matrix[:, 0], img_matrix[:, 1], img_matrix[:, 2], img_matrix[:, 3]

        p1 = np.array((a, center, b))
        p2 = np.array((b, center, c))
        p3 = np.array((c, center, d))
        p4 = np.array((d, center, a))

        plt.fill(p1[:, 0], p1[:, 1], "r")
        plt.fill(p2[:, 0], p2[:, 1], "g")
        plt.fill(p3[:, 0], p3[:, 1], "b")
        plt.fill(p4[:, 0], p4[:, 1], "y")

        # Clear grid
        plt.axis('off')
        plt.grid(b=None)
        plt.box(False)

    img.savefig(path.join(image_outpath, "{0}.png".format(image_count)), bbox_inches='tight', pad_inches=0)
    plt.close()


def fill_scene(max_shape_count: int, scene: Scene, json) -> None:

    scene_info = {"img_name": scene.name, "img_info": []}
    generated = 0
    failed = 0

    while failed < 10 and generated < max_shape_count:
        x = random.uniform(256, 768)
        y = random.uniform(256, 768)
        w = random.uniform(60, 160)
        h = random.uniform(60, 160)
        angle = random.uniform(0, 360)
        shape = Building(x, y, w, h, angle, generated)
        offset = np.array((random.uniform(0, w / 4), random.uniform(0, h / 4)))
        shape.offset_center(offset)

        if scene.has_overlap(shape):
            failed += 1

        else:
            scene.add_building(shape)
            building_data = {"building_id": generated}
            x_pts = shape.vertices[0,:].tolist()
            y_pts = shape.vertices[1,:].tolist()
            x_pts.append(shape.center[0])
            y_pts.append(shape.center[1])

            point_info = {"X": x_pts, "Y": y_pts}
            point_info["Edges"] = get_edges(point_info)
            point_info["Edges"].append(get_center_edges(point_info))
            point_info["Coarse Labels"] = get_coarse_labels(point_info)
            building_data["building_info"] = point_info
            scene_info["img_info"].append(building_data)
            generated += 1
            failed = 0

    json.append(scene_info)


def read_coordinates(data_dictionary):
    x_list = data_dictionary["X"]
    y_list = data_dictionary["Y"]
    return x_list, y_list


def get_edges(data_dictionary):

    x_list, y_list = read_coordinates(data_dictionary)
    edge_list = []

    cx = data_dictionary["X"][-1]
    cy = data_dictionary["Y"][-1]

    center = np.array((cx, cy))

    for i in range(0, len(x_list) - 1):
        anchor = np.array((x_list[i], y_list[i]))
        if i == 0:
            to_prev = len(x_list) - 2
            to_next = i + 1
        elif i == len(x_list) - 2:
            to_prev = i - 1
            to_next = 0

        edge_to_center = (center - anchor)
        edge_to_prev = (np.array((x_list[to_prev], y_list[to_prev])) - anchor)
        edge_to_next = (np.array((x_list[to_next], y_list[to_next])) - anchor)

        edge_to_center = np.arctan2(edge_to_center[1], edge_to_center[0])
        edge_to_prev = np.arctan2(edge_to_prev[1], edge_to_prev[0])
        edge_to_next = np.arctan2(edge_to_next[1], edge_to_next[0])

        edge_list.append([edge_to_prev, edge_to_center, edge_to_next])

    return edge_list


def get_center_edges(data_dictionary):
    x_list, y_list = read_coordinates(data_dictionary)
    cx = data_dictionary["X"][-1]
    cy = data_dictionary["Y"][-1]

    center = np.array((cx, cy))
    edges = []

    for i in range(len(x_list) - 1):
        edge = np.array((x_list[i], y_list[i]) - center)
        edge = np.arctan2(edge[1], edge[0])
        edges.append(edge)

    return edges


def get_coarse_labels(data_dictionary):
    coarse_list = []
    for edge_list in data_dictionary['Edges']:
        coarse_labels = [0] * 16
        for edge in edge_list:
            i = get_octant(edge)
            coarse_labels[i - 1] = 1
        coarse_list.append(coarse_labels)

    return coarse_list


def get_octant(angle) -> int:

    quadrant = math.pi / 8

    if 0 <= angle < quadrant:
        return 1
    elif quadrant <= angle < 2 * quadrant:
        return 2
    elif 2 * quadrant <= angle < 3 * quadrant:
        return 3
    elif 3 * quadrant <= angle <= 4 * quadrant:
        return 4
    elif 4 * quadrant <= angle < 5 * quadrant:
        return 5
    elif 5 * quadrant <= angle < 6 * quadrant:
        return 6
    elif 6 * quadrant <= angle < 7 * quadrant:
        return 7
    elif 7 * quadrant <= angle < 8 * quadrant:
        return 8
    elif -quadrant <= angle < 0:
        return 16
    elif -2 * quadrant <= angle < -quadrant:
        return 15
    elif -3 * quadrant <= angle < -2 * quadrant:
        return 14
    elif -4 * quadrant <= angle < -3 * quadrant:
        return 13
    elif -5 * quadrant <= angle < -4 * quadrant:
        return 12
    elif -6 * quadrant <= angle < -5 * quadrant:
        return 11
    elif -7 * quadrant <= angle < -6 * quadrant:
        return 10
    else:
        return 9

def plot_ground_truth(scene, image_outpath, image_count):

    truth = plt.figure(frameon=False)
    DPI = truth.get_dpi()
    truth.set_size_inches(1024.0 / float(DPI), 1024.0 / float(DPI))

    for structure in scene.buildings:
        x, y = structure.vertices[0], structure.vertices[1]
        plt.plot(x, y, 'o', color='black')
        plt.plot(structure.center[0], structure.center[1], 'o', color='black')

    axes = plt.gca()
    axes.set_xlim([0, 1024])
    axes.set_ylim([0, 1024])
    axes.axis('off')

    # Clear grid
    plt.axis('off')
    plt.grid(b=None)
    plt.box(False)

    truth.savefig(path.join(image_outpath, "image_{0}.png".format(image_count)), bbox_inches='tight', pad_inches=0)

    #buf = io.BytesIO()
    #truth.savefig(buf, format='png')
    #buf.seek(0)
    #im = Image.open(buf)
    #im.savefig
    #buf.close()

    plt.close()