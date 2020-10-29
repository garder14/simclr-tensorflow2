import tensorflow as tf



def nt_xent_loss(z, temperature):
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarity_matrix = tf.matmul(z, z, transpose_b=True)  # compute pairwise cosine similarities
    similarity_matrix_edit = tf.exp(similarity_matrix / temperature)  # divide by temperature and apply exp

    ij_indices = tf.reshape(tf.range(z.shape[0]), shape=[-1, 2])
    ji_indices = tf.reverse(ij_indices, axis=[1])
    positive_indices = tf.reshape(tf.concat([ij_indices, ji_indices], axis=1), shape=[-1, 2])  # indices of positive pairs: [[0, 1], [1, 0], [2, 3], [3, 2], ...]
    numerators = tf.gather_nd(similarity_matrix_edit, positive_indices)
    
    negative_mask = 1 - tf.eye(z.shape[0])  # mask that discards self-similarities
    denominators = tf.reduce_sum(tf.multiply(negative_mask, similarity_matrix_edit), axis=1)
    
    losses = -tf.math.log(numerators/denominators)
    return tf.reduce_mean(losses)
