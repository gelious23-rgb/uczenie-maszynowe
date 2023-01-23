from unittest import TestCase, main
from .. import Model as model


class ModelTest(TestCase):
    def test_predict_one(self):
        m = model.Model()
        m.load_trained(r'D:\py\ML_work\trained_models\model1.h5')
        self.assertEqual(m.predict_one(r'D:\py\ML_work\test_images\cats\cat.4003.jpg'), ('КОТ', 'D:\\py\\ML_work\\test_images\\cats\\cat.4003.jpg'))

    def test_predict_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as e:
            m = model.Model()
            m.load_trained(r'D:\py\ML_work\trained_models\model1.h5')
            m.predict_one(r'1.jpg')
        self.assertEqual(2, e.exception.args[0])

    def test_load_model(self):
        m = model.Model()
        m.load_trained(r'D:\py\ML_work\trained_models\model1.h5')
        self.assertIsNotNone(m._model)

    def test_load_model_no_file(self):
        with self.assertRaises(OSError) as e:
            m = model.Model()
            path = r'D:\py\ML\trained_models\model1.h5'
            m.load_trained(path)
        self.assertEqual(f'No file or directory found at {path}', e.exception.args[0])


if __name__ == '__main__':
    main()
