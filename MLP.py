import numpy as np
import tensorflow as tf
import time

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.models import Model

from parse_FEN import import_hash_table, FEN_to_input
from parsed_training_data import export_parsed_training_data, import_parsed_training_data


class MLP:
    """
    Creates a MLP neural network that predicts side advantage in a given chess position.
    This class compiles a keras model, as well as evaluates pre-existing models.
    To create training data, see stockfish_eval.py
    """
    def __init__(self,
                 model_name,
                 score_dict_file,
                 epochs=50,
                 batch_size=128,
                 activation_func='relu',
                 dropout=0.2,
                 num_classes=2):

        if not str(score_dict_file).endswith('.npy'):
            score_dict_file += '.npy'

        self.model_filename = model_name + '.model'
        self.score_dict_file = score_dict_file
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation_func = activation_func
        self.dropout = dropout
        self.num_classes = num_classes
        self.model = None

    def generate_training_data(self,
                               start=0,
                               end=-1,
                               min_move_number=0,
                               max_move_number=1000):
        train_boards = []
        train_attack_matrix = []
        train_scores = []
        counter = 0

        print('Importing FEN hash table...')
        score_dict = import_hash_table(self.score_dict_file)
        print('Hash table size: ' + str(len(score_dict)))
        binary_classification = self.num_classes == 2

        if end == -1:
            end = len(score_dict) - 1
        print('Generating training data for ' + str(end - start) + ' positions...')

        for FEN in score_dict:
            if counter >= end:
                break

            if counter >= start:
                # Convert FEN string to one hot encoded chess board
                board, attack_matrix, turn = FEN_to_input(FEN, min_move_number, max_move_number)

                if binary_classification:
                    # If white is winning, append 1 to labels. If black winning, append 0.
                    if board is not None:
                        score = score_dict[FEN]
                        if turn:
                            if score >= 0:
                                train_scores.append(1)
                            else:
                                train_scores.append(0)
                        else:
                            if score >= 0:
                                train_scores.append(0)
                            else:
                                train_scores.append(1)

                        train_boards.append(board)
                        train_attack_matrix.append(attack_matrix)
                else:
                    # If white is winning, append 1 to labels. If black winning, append 0, if draw append 2
                    if board is not None and attack_matrix is not None:
                        score = score_dict[FEN]
                        if turn:
                            if abs(score) <= 1.5:
                                train_scores.append([1, 0, 0])
                            elif score >= 0:
                                train_scores.append([0, 1, 0])
                            else:
                                train_scores.append([0, 0, 1])
                        else:
                            if abs(score) <= 1.5:
                                train_scores.append([1, 0, 0])
                            elif score >= 0:
                                train_scores.append([0, 0, 1])
                            else:
                                train_scores.append([0, 1, 0])

                        train_boards.append(board)
                        train_attack_matrix.append(attack_matrix)
            counter += 1

        train_boards = np.asarray(train_boards)
        train_attack_matrix = np.asarray(train_attack_matrix)
        train_scores = np.asarray(train_scores)
        return train_boards, train_attack_matrix, train_scores

    def train_model(self, import_training_data=None, export_training_data=None):
        # Generate one hot encoded boards with their respective engine evaluations
        start_time = time.time()

        # Instead of parsing training data, try to import pre-parsed data:
        if import_training_data is None:
            train_boards, train_attack_matrix, train_scores = self.generate_training_data()
            # train_attack_matrix = tf.keras.utils.normalize(train_attack_matrix)
        else:
            train_boards, train_attack_matrix, train_scores = import_parsed_training_data(str(import_training_data))

        if export_training_data is not None:
            export_parsed_training_data(train_boards, train_attack_matrix, train_scores, str(export_training_data))

        print('Training data finished (' + str(time.time() - start_time) + 's)')

        if self.num_classes == 2:
            self.binary_classifier()
        else:
            self.ternary_classifier()

        print('Training...')
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
        epoch_path = str(self.model_name) + '-{epoch:02d}-{val_acc:.2f}.model'
        checkpoint = ModelCheckpoint(epoch_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        self.model.fit([train_boards, train_attack_matrix], [train_scores],
                       epochs=self.epochs, validation_split=0.1, callbacks=[tensorboard, checkpoint])

        self.model.save(str(self.model_filename))

    def test_model(self, import_test_data=None, export_test_data=None):
        # Generate one hot encoded boards with their respective engine evaluations
        start_time = time.time()

        # Instead of parsing training data, try to import pre-parsed data:
        if import_test_data is None:
            test_boards, test_attack_matrix, test_scores = self.generate_training_data(min_move_number=15)
        else:
            test_boards, test_attack_matrix, test_scores = import_parsed_training_data(str(import_test_data))

        if export_test_data is not None:
            export_parsed_training_data(test_boards, test_attack_matrix, test_scores, str(export_test_data))

        print('Training data finished (' + str(time.time() - start_time) + 's)')

        model = tf.keras.models.load_model(self.model_filename)
        val_loss, val_acc = model.evaluate([test_boards, test_attack_matrix], [test_scores])
        print('Loss: ' + str(val_loss))
        print('Accuracy: ' + str(val_acc))

    def ternary_classifier(self):
        board_input = Input(shape=(768,), name='board_input')
        x = Dense(1024, activation=self.activation_func)(board_input)

        attack_matrix = Input(shape=(128,), name='attack_matrix')
        y = Dense(256, activation=self.activation_func)(attack_matrix)

        x = tf.keras.layers.concatenate([x, y])
        x = Dropout(self.dropout)(x)
        x = Dense(1024, activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)
        x = Dense(512, activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)
        x = Dense(50, activation=self.activation_func)(x)
        main_output = Dense(3, activation='softmax', name='main_output')(x)

        self.model = Model(inputs=[board_input], outputs=[main_output])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def binary_classifier(self):
        board_input = Input(shape=(768,), name='board_input')
        x = Dense(1024, activation=self.activation_func)(board_input)

        attack_matrix = Input(shape=(128,), name='attack_matrix')
        y = Dense(256, activation=self.activation_func)(attack_matrix)

        x = tf.keras.layers.concatenate([x, y])
        x = Dropout(self.dropout)(x)
        x = Dense(1280, activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)
        x = Dense(512, activation=self.activation_func)(x)
        x = Dropout(self.dropout)(x)
        x = Dense(50, activation=self.activation_func)(x)
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)

        self.model = Model(inputs=[board_input, attack_matrix], outputs=[main_output])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])


if __name__ == '__main__':
    # NN = MLP('binary_model_new_attack_matv2', 'known_scores(2.7).npy', num_classes=2, activation_func='elu')
    # NN = MLP('binary_model_new_attack_mat', 'train_set(3000000).npy', num_classes=2, activation_func='relu')
    # NN.train_model(export_training_data='binary_data_128attack_mat.npy')
    # NN.train_model(export_training_data='binary_data_128attack_mat_3000000.npy')
    # NN.train_model(import_training_data='binary_data_128attack_mat_3000000.npy')
    # NN_tester = MLP('binary_model_new_attack_mat', 'test_set(666438).npy', num_classes=2)
    NN_tester = MLP('chess_model_no_attack_matv1-20-0.89', 'test_set(666438).npy', num_classes=2)
    NN_tester.test_model()
    # NN.train_model(export_training_data='ternary_parsed_data.npy')
    # NN.train_model(export_training_data='binary_classification_data_no_attack_mat.npy')
    # NN.train_model(import_training_data='binary_classification_data_no_attack_mat.npy')
    # NN.train_model()
    # NN.train_model(import_training_data='ternary_parsed_data.npy')
    # NN.train_model(import_training_data='parsed_training_data.npy')
