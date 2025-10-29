import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, n_node, node_emd_init, config):
        super(Generator, self).__init__()
        self.n_node = n_node
        self.emd_dim = config.n_emb
        self.node_emd_init = node_emd_init
        self.config = config

        initializer = tf.keras.initializers.GlorotNormal()

        # Create model variables
        if node_emd_init is not None:
            self.node_embedding_matrix = self.add_weight(
                name='gen_node_embedding',
                shape=self.node_emd_init.shape,
                initializer=tf.constant_initializer(self.node_emd_init),
                trainable=True)
        else:
            self.node_embedding_matrix = self.add_weight(
                name='gen_node_embedding',
                shape=[self.n_node, self.emd_dim],
                initializer=initializer,
                trainable=True)

        self.gen_w_1 = self.add_weight(
            name='gen_w',
            shape=[2, self.emd_dim, self.emd_dim],
            initializer=initializer,
            trainable=True)

        self.gen_b_1 = self.add_weight(
            name='gen_b',
            shape=[2, self.emd_dim],
            initializer=initializer,
            trainable=True)

        self.gen_w_2 = self.add_weight(
            name='gen_w_2',
            shape=[2, self.emd_dim, self.emd_dim],
            initializer=initializer,
            trainable=True)

        self.gen_b_2 = self.add_weight(
            name='gen_b_2',
            shape=[2, self.emd_dim],
            initializer=initializer,
            trainable=True)

    def generate_node(self, node_embedding, noise_embedding, direction):
        """Generate embeddings for a node in a specific direction"""
        input_tensor = tf.reshape(node_embedding, [-1, self.emd_dim])
        input_tensor = input_tensor + noise_embedding
        output = tf.nn.leaky_relu(tf.matmul(input_tensor, self.gen_w_1[direction]) + self.gen_b_1[direction])
        return output

    def call(self, inputs, training=None):
        """Forward pass of the generator"""
        node_ids, noise_embedding, dis_node_embedding = inputs
        
        # Get node embeddings
        node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, node_ids)
        
        # Process embeddings for source and target directions
        _noise_embedding = []
        for i in range(2):
            _noise_embedding.append(tf.reshape(
                tf.gather(noise_embedding, [i], axis=0), 
                [-1, self.emd_dim]))

        _dis_node_embedding = []
        for i in range(2):
            _dis_node_embedding.append(tf.reshape(
                tf.gather(dis_node_embedding, [i], axis=0), 
                [-1, self.emd_dim]))

        _neg_loss = [0.0, 0.0]
        _fake_node_embedding_list = []
        _score = [0, 0]

        for i in range(2):
            _fake_node_embedding = self.generate_node(node_embedding, _noise_embedding[i], i)
            _fake_node_embedding_list.append(_fake_node_embedding)

            _score[i] = tf.reduce_sum(tf.multiply(_dis_node_embedding[i], _fake_node_embedding), axis=1)

            _neg_loss[i] = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(_score[i]) * (1.0 - self.config.label_smooth),
                    logits=_score[i])) \
                + self.config.lambda_gen * (tf.nn.l2_loss(node_embedding) + tf.nn.l2_loss(self.gen_w_1[i]))
                
        return {
            'neg_loss': _neg_loss,
            'fake_node_embedding': _fake_node_embedding_list,
            'loss': _neg_loss[0] + _neg_loss[1]
        }