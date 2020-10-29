import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from datasets import CIFAR10
from models import ResNet18, ResNet34, ProjectionHead
from losses import nt_xent_loss



encoders = {'resnet18': ResNet18, 'resnet34': ResNet34}


def main(args):

    # Load CIFAR-10 dataset
    data = CIFAR10()

    # Instantiate networks f and g
    f_net = encoders[args.encoder]()
    g_net = ProjectionHead()

    # Initialize the weights of f and g
    x = data.get_batch_pretraining(batch_id=0, batch_size=args.batch_size)
    h = f_net(x, training=False)
    print('Shape of h:', h.shape)
    z = g_net(h, training=False)
    print('Shape of z:', z.shape)

    num_params_f = tf.reduce_sum([tf.reduce_prod(var.shape) for var in f_net.trainable_variables])    
    print('Networks f ({} trainable parameters) and g initialized.'.format(num_params_f))


    # Define optimizer
    lr = 1e-3 * args.batch_size / 512
    opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=1e-6)
    print('Using Adam optimizer with learning rate {} and weight decay 1e-6.'.format(lr))


    @tf.function
    def train_step_pretraining(x):  # (2*bs, 32, 32, 3)

        # Forward pass
        with tf.GradientTape(persistent=True) as tape:
            h = f_net(x, training=True)  # (2*bs, 512)
            z = g_net(h, training=True)  # (2*bs, 128)
            loss = nt_xent_loss(z, temperature=tf.constant(args.temperature))
        
        # Backward pass
        grads = tape.gradient(loss, f_net.trainable_variables)
        opt.apply_gradients(zip(grads, f_net.trainable_variables))
        grads = tape.gradient(loss, g_net.trainable_variables)
        opt.apply_gradients(zip(grads, g_net.trainable_variables))
        del tape

        return loss


    # Train f and g to minimize NT-Xent loss
    batches_per_epoch = data.num_train_images // args.batch_size

    log_every = 10  # batches
    save_every = 100  # epochs

    losses = []
    for epoch_id in range(args.num_epochs):
        data.shuffle_training_data()
        
        for batch_id in range(batches_per_epoch):
            x = data.get_batch_pretraining(batch_id, args.batch_size)
            loss = train_step_pretraining(x)
            losses.append(float(loss))
            if (batch_id + 1) % log_every == 0:
                print('[Epoch {}/{} Batch {}/{}] Loss={:.4f}.'.format(epoch_id+1, args.num_epochs, batch_id+1, batches_per_epoch, loss))
        
        if (epoch_id + 1) % save_every == 0:
            f_net.save_weights('f{}.h5'.format(epoch_id+1))
            print('Weights of f saved.')
    
    np.savetxt('losses.txt', tf.stack(losses).numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder', type=str, required=True, choices=['resnet18', 'resnet34'], help='Encoder architecture')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for pretraining')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for NT-Xent loss')

    args = parser.parse_args()
    main(args)
