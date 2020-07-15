from utils.plot_functions import fill_scene, plot_sample_img, plot_ground_truth
from utils.clean_training import clean_json_labels, clean_json_points
from Structure.Scene import Scene
import json

path_to_training_images = 'E:Images\Synthesized_Data\Dataset_Sample\Image'
path_to_training_images1 = 'E:/Amin_Code/Position_feature/Image'
path_to_training_images2 = 'E:/Amin_Code/PositionOnly/Image'
path_to_training_truth = 'E:\Images\Synthesized_Data\Dataset_Sample\BuildingSegment'
path_to_training_json = 'C:/Users/Brian/PycharmProjects/SyntheticDataGenerator/json/training.json'

path_to_validation_images = 'E:\Images\Synthesized_Data\Validation'
path_to_validation_truth = ''
path_to_validation_json = 'C:/Users/Brian/PycharmProjects/SyntheticDataGenerator/json/validation.json'


path_to_testing_images = 'E:\Images\Synthesized_Data\Testing'
path_to_testing_truth = ''
path_to_testing_json = 'C:/Users/Brian/PycharmProjects/SyntheticDataGenerator/json/testing.json'

training_json = []
validation_json = []
testing_json = []

noise = False

training_sample_size = 100
validation_sample_size = training_sample_size // 5
testing_sample_size = training_sample_size // 5

scene_size = (128, 128)

for i in range(training_sample_size):
    curr_scene_id = "{0}".format(i)
    curr_scene_id = curr_scene_id.zfill(10)
    scene = Scene(scene_size, curr_scene_id)
    max_buildings = 12
    fill_scene(max_buildings, scene, training_json)
    plot_sample_img(scene, path_to_training_images1, i, add_noise=noise)
    plot_sample_img(scene, path_to_training_images2, i, add_noise=noise)
    plot_ground_truth(scene, path_to_training_truth, i)

with open(path_to_training_json, 'w') as outfile:
    json.dump(training_json, outfile)

clean_json_points(training_json, 'E:/Amin_Code/Position_feature/Point_Location')
clean_json_labels(training_json, 'E:/Amin_Code/Position_feature/Coarse_Label')
clean_json_points(training_json, 'E:/Amin_Code/PositionOnly/Point_Location')
clean_json_labels(training_json, 'E:/Amin_Code/PositionOnly/Coarse_Label')

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