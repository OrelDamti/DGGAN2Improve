import tensorflow as tf

class Discriminator(tf.keras.Model):
    def __init__(self, n_node, node_emd_init, config):
        super(Discriminator, self).__init__()
        self.n_node = n_node
        self.emd_dim = config.n_emb
        self.node_emd_init = node_emd_init
        self.config = config

        initializer = tf.keras.initializers.GlorotNormal()

        # Create model variables
        if node_emd_init is not None:
            self.node_embedding_matrix = self.add_weight(
                name='dis_node_embedding',
                shape=self.node_emd_init.shape,
                initializer=tf.constant_initializer(self.node_emd_init),
                trainable=True)
        else:
            self.node_embedding_matrix = self.add_weight(
                name='dis_node_embedding',
                shape=[2, self.n_node, self.emd_dim],
                initializer=initializer,
                trainable=True)

    def call(self, inputs, training=None):
        """Forward pass of the discriminator"""
        pos_node_ids, pos_node_neighbor_ids, fake_node_embedding = inputs
        
        # Get node embeddings for both directions
        _node_embedding_matrix = []
        for i in range(2):
            _node_embedding_matrix.append(tf.reshape(
                tf.gather(self.node_embedding_matrix, [i], axis=0),
                [-1, self.emd_dim]))

        # Get embeddings for positive samples
        pos_node_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[0], pos_node_ids)
        pos_node_neighbor_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[1], pos_node_neighbor_ids)

        # Calculate positive loss
        pos_score = tf.matmul(pos_node_embedding, pos_node_neighbor_embedding, transpose_b=True)
        pos_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(pos_score), logits=pos_score))

        # Calculate negative losses
        _neg_loss = [0, 0, 0, 0]
        node_id = [pos_node_ids, pos_node_neighbor_ids]
        
        for i in range(2):
            for j in range(2):
                node_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[j], node_id[i])
                _fake_node_embedding = tf.reshape(
                    tf.gather(fake_node_embedding, [i], axis=0), [2, -1, self.emd_dim])
                _fake_node_embedding = tf.reshape(
                    tf.gather(_fake_node_embedding, [j], axis=0), [-1, self.emd_dim])

                neg_score = tf.matmul(node_embedding, _fake_node_embedding, transpose_b=True)
                _neg_loss[i * 2 + j] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(neg_score), logits=neg_score))

        # Calculate total loss
        total_loss = pos_loss + \
                    _neg_loss[0] * self.config.neg_weight[0] + \
                    _neg_loss[1] * self.config.neg_weight[1] + \
                    _neg_loss[2] * self.config.neg_weight[2] + \
                    _neg_loss[3] * self.config.neg_weight[3]

        return {
            'pos_loss': pos_loss,
            'neg_loss': _neg_loss,
            'loss': total_loss
        }