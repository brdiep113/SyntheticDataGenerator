from utils.plot_functions import fill_scene, plot_sample_img, plot_ground_truth
from Structure.Scene import Scene
import json

path_to_training_images = 'E:Images\Synthesized_Data\Training'
path_to_training_truth = 'E:\Images\Synthesized_Data'
path_to_training_json = 'C:/Users/Brian/PycharmProjects/SyntheticDataGenerator/json'

path_to_validation_images = 'E:\Images\Synthesized_Data\Validation'
path_to_validation_truth = ''
path_to_validation_json = 'C:/Users/Brian/PycharmProjects/SyntheticDataGenerator/json'


path_to_testing_images = 'E:\Images\Synthesized_Data\Testing'
path_to_testing_truth = ''
path_to_testing_json = 'C:/Users/Brian/PycharmProjects/SyntheticDataGenerator/json'

training_json = []
validation_json = []
testing_json = []

training_sample_size = 10
validation_sample_size = training_sample_size // 5
testing_sample_size = training_sample_size // 5

scene_size = (1024, 1024)

for i in range(training_sample_size):
    curr_scene_id = "img{0}".format(i)
    scene = Scene(scene_size, curr_scene_id)
    max_buildings = 24
    fill_scene(max_buildings, scene, training_json)
    plot_sample_img(scene, path_to_training_images, i)
    #plot_ground_truth(scene, path_to_training_truth, i)

with open(path_to_training_json, 'w') as outfile:
    json.dump(training_json, outfile)

for i in range(validation_sample_size):
    curr_scene_id = "img{0}".format(i)
    scene = Scene(scene_size, curr_scene_id)
    max_buildings = 24
    fill_scene(max_buildings, scene, validation_json)
    plot_sample_img(scene, path_to_validation_images, i)
    #plot_ground_truth(scene, path_to_validation_truth, i)

with open(path_to_validation_json, 'w') as outfile:
    json.dump(validation_json, outfile)

for i in range(testing_sample_size):
    curr_scene_id = "test_img{0}".format(i)
    scene = Scene(scene_size, curr_scene_id)
    max_buildings = 24
    fill_scene(max_buildings, scene, testing_json)
    plot_sample_img(scene, path_to_testing_images, i)
    #plot_ground_truth(scene, path_to_testing_truth, i)

with open(path_to_testing_json, 'w') as outfile:
    json.dump(training_json, outfile)