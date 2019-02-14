import os
from keras.utils.vis_utils import plot_model

def SaveModel(model, store_folder):
    plot_model(model, to_file=os.path.join(store_folder, 'model.png'), show_shapes=True)
    model.summary()
    model_yaml = model.to_yaml()
    with open(os.path.join(store_folder, 'model.yaml'), "w") as yaml_file:
        yaml_file.write(model_yaml)