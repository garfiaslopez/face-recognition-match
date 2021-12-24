import os
import turicreate as tc

data = tc.image_analysis.load_images('photos/single-augmented', with_path=True)
data['label'] = data['path'].apply(lambda path: os.path.basename(os.path.dirname(path)))
model = tc.image_classifier.create(data, target='label', model='resnet-50', max_iterations=100)
model.export_coreml('models/face_recognition_single_augmented.mlmodel')