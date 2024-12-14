# model_builder.py
import numpy as np
import tensorflow as tf
from keras import Model, layers
from tensorflow.keras import backend as K

class PositionalEncodingLayer(layers.Layer):
    # 如之前定义的PositionalEncoding, 如果在该模型中不需要位置编码可略去
    def __init__(self, seq_len, d_model):
        super(PositionalEncodingLayer, self).__init__()
        self.pos_encoding = self.get_positional_encoding(seq_len, d_model)
    
    def get_positional_encoding(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

class AttentionLayer(layers.Layer):
    """
    简单加性注意力（Bahdanau attention）
    输入:
        query: (batch, hidden_dim) 来自LSTM最后时刻的隐状态
        values: (batch, timesteps, hidden_dim) 来自LSTM所有时刻的输出
    输出:
        context_vector: 加权求和后的上下文向量 (batch, hidden_dim)
        attention_weights: (batch, timesteps)
    """
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        # query (batch, hidden_dim)
        # values (batch, timesteps, hidden_dim)
        query_with_time_axis = tf.expand_dims(query, 1)
        # (batch, timesteps, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        # attention_weights shape: (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum: (batch, hidden_dim)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        attention_weights = tf.squeeze(attention_weights, axis=-1)
        return context_vector, attention_weights

class TransformerModelBuilder:
    # 保留原先Transformer相关定义，以便在需要时调用
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(inputs + x)

        x_ff = layers.Dense(ff_dim, activation='relu')(x)
        x_ff = layers.Dropout(dropout)(x_ff)
        x_ff = layers.Dense(inputs.shape[-1])(x_ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + x_ff)
        return x

    def build_transformer_model_with_pid(self, input_shape, pid_vocab_size, forecast_horizon, target_len, pid_embedding_dim=10, head_size=64, num_heads=4, ff_dim=128, num_transformer_blocks=2, dropout=0.1):
        features_input = layers.Input(shape=input_shape, name='features')
        pid_input = layers.Input(shape=(1,), name='pid')

        pid_embedding = layers.Embedding(input_dim=pid_vocab_size, output_dim=pid_embedding_dim, name='pid_embedding')(pid_input)
        pid_embedding = layers.Flatten()(pid_embedding)

        x = PositionalEncodingLayer(seq_len=input_shape[0], d_model=input_shape[1])(features_input)

        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)

        x = layers.Concatenate()([x, pid_embedding])
        outputs = layers.Dense(forecast_horizon * target_len)(x)
        outputs = layers.Reshape((forecast_horizon, target_len))(outputs)
        model = Model(inputs=[features_input, pid_input], outputs=outputs)
        return model

    def wmape(self, y_true, y_pred):
        epsilon = K.epsilon()
        return K.sum(K.abs(y_true - y_pred)) / (K.sum(K.abs(y_true)) + epsilon)

    def build_lstm_attention_model_with_pid(self, input_shape, pid_vocab_size, forecast_horizon, target_len, pid_embedding_dim=10, lstm_units=64, attention_units=64, dropout=0.1):
        # 输入
        features_input = layers.Input(shape=input_shape, name='features')
        pid_input = layers.Input(shape=(1,), name='pid')

        # PID嵌入
        pid_embedding = layers.Embedding(input_dim=pid_vocab_size, output_dim=pid_embedding_dim, name='pid_embedding')(pid_input)
        pid_embedding = layers.Flatten()(pid_embedding)

        # LSTM层，返回序列输出和最终隐藏状态
        lstm_output, state_h, state_c = layers.LSTM(lstm_units, return_sequences=True, return_state=True)(features_input)
        
        # Attention层，对LSTM所有时刻输出lstm_output进行加权
        context_vector, attention_weights = AttentionLayer(attention_units)(state_h, lstm_output)

        # 合并context_vector与LSTM最后状态
        x = layers.Concatenate()([context_vector, state_h])  
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout)(x)

        # 合并PID嵌入
        x = layers.Concatenate()([x, pid_embedding])

        # 输出
        outputs = layers.Dense(forecast_horizon * target_len)(x)
        outputs = layers.Reshape((forecast_horizon, target_len))(outputs)

        model = Model(inputs=[features_input, pid_input], outputs=outputs)
        return model
