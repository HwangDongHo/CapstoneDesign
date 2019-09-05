from src.align.Align_mtcnn import get_align
from src.train_softmax import train_softmax
from src.classifier import classification
from src.freeze_graph import make_pb

import facenet


get_align('src/align/test_0823', 'src/align/test_0823_align', image_size=182, margin=44, random_order='store_true', gpu_memory_fraction=1.0, detect_multiple_faces=False)
#train_softmax()
make_pb('src/models/facenet/20190823-191957', 'src/models/facenet/20190823-191957/output.pb')
#classification('src/align/test_0823_align', 'src/models/facenet/20190823-191957/20190823-191957.pb', 'src/models/facenet/20190823-191957/lfw_classifier_333.pkl', batch_size=90, seed=666, image_size=160)