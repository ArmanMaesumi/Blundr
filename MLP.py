import numpy as np
import tensorflow as tf
import time

from generate_training_data import TrainingDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.python.keras.models import Model


class MLP:
    """
    Creates an MLP neural network that predicts side advantage in a given chess position.
    This class compiles a keras model, as well as evaluates pre-existing models.
    To create training data, see stockfish_eval.py
    """
    def __init__(self,
                 model_name,
                 score_dict_file,
                 pretrained_weights,
                 epochs=200,
                 batch_size=128,
                 activation_func='relu',
                 dropout=0.2,
                 num_classes=2):

        np.random.seed(42)
        if not str(score_dict_file).endswith('.npy'):
            score_dict_file += '.npy'

        self.model_filename = model_name + '.model'
        self.score_dict_file = score_dict_file
        self.model_name = model_name
        self.pretrained_weights = pretrained_weights
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.dropout = dropout
        self.num_classes = num_classes
        self.model = None

    # def generate_training_data(self,
    #                            start=0,
    #                            end=-1,
    #                            min_move_number=0,
    #                            max_move_number=1000):
    #     train_boards = []
    #     train_attack_matrix = []
    #     train_scores = []
    #     counter = 0
    #
    #     print('Importing FEN hash table...')
    #     score_dict = import_hash_table(self.score_dict_file)
    #     print('Hash table size: ' + str(len(score_dict)))
    #     binary_classification = self.num_classes == 2
    #
    #     if end == -1:
    #         end = len(score_dict) - 1
    #     print('Generating training data for ' + str(end - start) + ' positions...')
    #
    #     for FEN in score_dict:
    #         if counter >= end:
    #             break
    #
    #         if counter >= start:
    #             # Convert FEN string to one hot encoded chess board
    #             board, attack_matrix, turn = FEN_to_input(FEN, min_move_number, max_move_number)
    #
    #             if binary_classification:
    #                 # If white is winning, append 1 to labels. If black winning, append 0.
    #                 if board is not None:
    #                     score = score_dict[FEN]
    #                     if turn:
    #                         if score >= 0:
    #                             train_scores.append(1)
    #                         else:
    #                             train_scores.append(0)
    #                     else:
    #                         if score >= 0:
    #                             train_scores.append(0)
    #                         else:
    #                             train_scores.append(1)
    #
    #                     train_boards.append(board)
    #                     train_attack_matrix.append(attack_matrix)
    #             else:
    #                 # If white is winning, append 1 to labels. If black winning, append 0, if draw append 2
    #                 if board is not None and attack_matrix is not None:
    #                     score = score_dict[FEN]
    #                     if turn:
    #                         if abs(score) <= 1.5:
    #                             train_scores.append([1, 0, 0])
    #                         elif score >= 0:
    #                             train_scores.append([0, 1, 0])
    #                         else:
    #                             train_scores.append([0, 0, 1])
    #                     else:
    #                         if abs(score) <= 1.5:
    #                             train_scores.append([1, 0, 0])
    #                         elif score >= 0:
    #                             train_scores.append([0, 0, 1])
    #                         else:
    #                             train_scores.append([0, 1, 0])
    #
    #                     train_boards.append(board)
    #                     train_attack_matrix.append(attack_matrix)
    #         counter += 1
    #
    #     train_boards = np.asarray(train_boards)
    #     train_attack_matrix = np.asarray(train_attack_matrix)
    #     train_scores = np.asarray(train_scores)
    #     return train_boards, train_attack_matrix, train_scores

    def train_model(self, import_training_data=None, export_training_data=None):
        start_time = time.time()

        data_generator = TrainingDataGenerator(self.score_dict_file, num_classes=self.num_classes)
        train_boards, train_scores = data_generator.parse_FEN_dict()

        # Instead of parsing training data, try to import pre-parsed data:
        # if import_training_data is None:
        #     data_generator = TrainingDataGenerator(self.score_dict_file, num_classes=self.num_classes)
        #     train_boards, train_scores = data_generator.parse_FEN_dict()
        # else:
        #     train_boards, train_attack_matrix, train_scores = import_parsed_training_data(str(import_training_data))
        #
        # if export_training_data is not None:
        #     export_parsed_training_data(train_boards, train_attack_matrix, train_scores, str(export_training_data))

        print('Training data finished (' + str(time.time() - start_time) + 's)')

        if self.num_classes == 2:
            self.binary_classifier()
        else:
            self.ternary_classifier()

        if self.pretrained_weights is not None:
            print('Loading weights from ' + str(self.pretrained_weights))
            self.model = tf.keras.models.load_model(self.pretrained_weights)
            self.model.load_weights(self.pretrained_weights)
            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

        print('Training...')
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
        epoch_path = str(self.model_name) + '-{epoch:02d}-{val_acc:.2f}.model'
        # checkpoint = ModelCheckpoint(epoch_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        checkpoint = ModelCheckpoint(epoch_path, period=20)
        self.model.fit([train_boards], [train_scores],
                       epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.10, callbacks=[tensorboard, checkpoint])

        self.model.save(str(self.model_filename))

    def test_model(self, import_test_data=None, export_test_data=None):
        start_time = time.time()

        data_generator = TrainingDataGenerator(self.score_dict_file, num_classes=self.num_classes)
        test_boards, test_scores = data_generator.parse_FEN_dict()

        # # Instead of parsing training data, try to import pre-parsed data:
        # if import_test_data is None:
        #     # test_boards, test_attack_matrix, test_scores = self.generate_training_data(min_move_number=15)
        #     data_generator = TrainingDataGenerator(self.score_dict_file, num_classes=self.num_classes)
        #     test_boards, test_scores = data_generator.parse_FEN_dict()
        # else:
        #     test_boards, test_attack_matrix, test_scores = import_parsed_training_data(str(import_test_data))
        #
        # if export_test_data is not None:
        #     export_parsed_training_data(test_boards, test_attack_matrix, test_scores, str(export_test_data))

        print('Training data finished (' + str(time.time() - start_time) + 's)')
        model = tf.keras.models.load_model(self.model_filename)
        loss, acc = model.evaluate([test_boards], [test_scores])
        print('Loss: ' + str(loss))
        print('Accuracy: ' + str(acc))

    # Classify White Win/Black Win/Draw states
    def ternary_classifier(self):
        board_input = Input(shape=((64 * 12) + 3,), name='board_input')

        x = Dense(1048)(board_input)
        # x = BatchNormalization()(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        x = Dense(500)(x)
        # x = BatchNormalization()(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        x = Dense(50)(x)
        # x = BatchNormalization()(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        main_output = Dense(3, name='main_output')(x)
        main_output = Activation(activation='softmax')(main_output)

        self.model = Model(inputs=[board_input], outputs=[main_output])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    # Classify White Win/Black Win states
    def binary_classifier(self):
        board_input = Input(shape=((64*12) + 3,), name='board_input')

        x = Dense(1024)(board_input)
        x = BatchNormalization()(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        x = Dense(512)(x)
        x = BatchNormalization()(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation(activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)

        main_output = Dense(1, name='main_output')(x)
        main_output = BatchNormalization()(main_output)
        main_output = Activation(activation='sigmoid')(main_output)

        self.model = Model(inputs=[board_input], outputs=[main_output])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])


if __name__ == '__main__':
    NN = MLP('ternary_relu_adam_combineddataset', 'combined(5000000).npy',
             num_classes=3, activation_func='relu',
             pretrained_weights=None)

    NN.train_model()

    # NN = MLP('binary_model_new_attack_matv2', 'known_scores(2.7).npy', num_classes=2, activation_func='elu')
    # NN = MLP('binary_model_new_attack_mat', 'train_set(3000000).npy', num_classes=2, activation_func='relu')
    # NN.train_model(export_training_data='binary_data_128attack_mat.npy')
    # NN.train_model(export_training_data='binary_data_128attack_mat_3000000.npy')
    # NN.train_model(import_training_data='binary_data_128attack_mat_3000000.npy')
    # NN_tester = MLP('binary_model_new_attack_mat', 'test_set(666438).npy', num_classes=2)
    # NN = MLP('ternary_relu_adam_50_draw150_balanced-40-0.90', 'test_set(300000)_FICS.npy',
    #          num_classes=3, activation_func='relu',
    #          pretrained_weights=None)
    # NN = MLP('binary_turncheck_relu_adam_batchnorm_256_attackmat_FICS-42-0.91', 'test_set(300000)_FICS.npy',
    #          num_classes=2, activation_func='relu',
    #          pretrained_weights=None)
    # NN = MLP('ternary_turncheck_relu_adam_FICS-21-0.85', 'test_set(300000)_FICS.npy', num_classes=3, activation_func='relu')
    # NN = MLP('new_board_rep_binary_adam_relu_norm', 'train_set(3500000).npy', num_classes=2, activation_func='relu')
    # NN_tester = MLP('new_board_rep-10-0.86', 'test_set(666438).npy', num_classes=2)
    # NN_tester.test_model()
    # NN.train_model(export_training_data='ternary_check2_balanced(3000000).npy')
    #NN_tester = MLP('chess_model_no_attack_matv1-20-0.89', 'test_set(666438).npy', num_classes=2)
    #NN_tester.test_model()
    # NN.train_model(export_training_data='ternary_parsed_data.npy')
    # NN.train_model(export_training_data='binary_classification_data_no_attack_mat.npy')
    # NN.train_model(import_training_data='binary_classification_data_no_attack_mat.npy')
    # NN.train_model()
    # NN.train_model(import_training_data='ternary_parsed_data.npy')
    # NN.train_model(import_training_data='parsed_training_data.npy')
