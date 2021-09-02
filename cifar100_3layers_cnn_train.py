import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def plot_result(history):
    '''
    plot result
    全ての学習が終了した後に、historyを参照して、accuracyとlossをそれぞれplotする
    '''

    # accuracy
    plt.figure()
    plt.plot(history.history['acc'], label='acc', marker='.')
    plt.plot(history.history['val_acc'], label='val_acc', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('accuracy')
    fig_name = os.path.splitext(os.path.basename(__file__))[0] + '_accuracy.png'
    plt.savefig(fig_name)

    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='loss', marker='.')
    plt.plot(history.history['val_loss'], label='val_loss', marker='.')
    plt.grid()
    plt.legend(loc='best')
    plt.title('loss')
    fig_name = os.path.splitext(os.path.basename(__file__))[0] + '_loss.png'
    plt.savefig(fig_name)


def create_model_three_layers(num_classes, input_shape):
    """
    モデル作成(3層)

    Parameters
    ----------
    num_classes : int
        クラス数
    input_shape : tuple
        入力画像サイズ
    """
    inputs = x = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(32,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.Conv2D(32,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.Conv2D(128,3,padding='same',kernel_initializer='he_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes,kernel_regularizer=tf.keras.regularizers.l2(1e-4),activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model

def _main(num_data, input_shape, num_classes, epochs, batch_size):
    """
    cifar100学習プログラム
        
    Parameters
    ----------
    num_data : int
        学習する画像枚数の指定
    input_shape: tuple
        入力画像の形状
    num_classes : int
        クラス数
    epochs : int
        エポック数
    batch_size : int
        バッチサイズ
    """
    
    # 学習データ読込(cifar10)
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    
    # データ件数を絞る
    X_train = X_train[:num_data]
    y_train = y_train[:num_data]
    print(X_train.shape)

    # データの前処理
    X_train = np.array(X_train, np.float32) / 255
    X_test = np.array(X_test, np.float32) / 255

    # 正解ラベルを'one hot表現'に変形
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # 学習モデル作成
    model = create_model_three_layers(num_classes, input_shape=input_shape)

    # 最適化アルゴリズム
    optimizer = tf.keras.optimizers.Adam()

    # モデルの設定
    model.compile(
        loss='categorical_crossentropy', # 損失関数の設定
        optimizer=optimizer, # 最適化法の指定
        metrics=['acc'])

    # モデル情報表示
    model.summary()

    # モデルの学習
    history = model.fit(X_train,
              y_train,
              validation_data=(X_test, y_test),
              batch_size=batch_size,
              epochs=epochs)

    plot_result(history)

    # モデル保存
    model.save('model_cifar100_3layers_cnn.h5')

    # 評価 & 評価結果出力
    score = model.evaluate(X_test, y_test)
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

# エントリポイント
if __name__ == "__main__":
    # データ枚数
#    num_data = 60000
    num_data = 2000
    # 入力画像の形状
    input_shape = (32, 32, 3)
    # 分類クラス数
    num_classes = 100
    # エポック数
    epochs = 30
    # バッチサイズ
    batch_size = 128
    _main(num_data, input_shape, num_classes, epochs, batch_size)
