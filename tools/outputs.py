import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .utils import predict


##
def print_out(args, sess, env, X_adv, X_test, y_test):

    print('\nRandomly sample adversarial data from each category')

    y1 = predict(sess, env, X_test)
    y2 = predict(sess, env, X_adv)

    z0 = np.argmax(y_test, axis=1)
    z1 = np.argmax(y1, axis=1)
    z2 = np.argmax(y2, axis=1)

    X_tmp = np.empty((args.n_classes, args.img_size, args.img_size, args.img_chan))
    X_tmp = np.squeeze(X_tmp)
    y_tmp = np.empty((args.n_classes, args.n_classes))
    for i in range(args.n_classes):
        print('Target {0}'.format(i))
        ind, = np.where(np.all([z0 == i, z1 == i, z2 != i], axis=0))
        cur = np.random.choice(ind)
        X_tmp[i] = np.squeeze(X_adv[cur])
        y_tmp[i] = y2[cur]

    print('\nPlotting results')

    fig = plt.figure(figsize=(args.n_classes, 1.2))
    gs = gridspec.GridSpec(1, args.n_classes, wspace=0.05, hspace=0.05)

    label = np.argmax(y_tmp, axis=1)
    proba = np.max(y_tmp, axis=1)
    for i in range(args.n_classes):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X_tmp[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(label[i], proba[i]),
                    fontsize=12)

    print('\nSaving figure')

    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/{0}_{1}.png'.format(args.dataset, args.attack))


##
def print_sample(args, sess, env, X_adv, X_test, y_test):
    """ len(X_adv) <= 10 で使用 """

    if args.dataset == 'mnist':
        labels = ['0','1','2','3','4','5','6','7','8','9']
    elif args.dataset == 'cifar10':
        labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    y_test_label = np.argmax(y_test, axis=1)
    y_test_proba = np.max(y_test, axis=1)
    
    y_adv = predict(sess, env, X_adv)
    y_adv_label = np.argmax(y_adv, axis=1)
    y_adv_proba = np.max(y_adv, axis=1)
    
    fig = plt.figure(figsize=(10*2, 4))
    gs = gridspec.GridSpec(2, 10, wspace=0.05, hspace=0.05)
    
    # for image
    X_test = np.squeeze(X_test)
    X_adv = np.squeeze(X_adv)
    for i in range(len(X_adv)):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X_test[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(labels[y_test_label[i]], y_test_proba[i]),
                    fontsize=15)

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(X_adv[i], cmap='gray', interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('{0} ({1:.2f})'.format(labels[y_adv_label[i]], y_adv_proba[i]),
                    fontsize=15)

    gs.tight_layout(fig)
    os.makedirs('img', exist_ok=True)
    plt.savefig('img/sample[{0}-{1}]({2}).png'.format(args.start, args.stop, args.epsilon))

