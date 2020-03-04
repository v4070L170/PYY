import os
import argparse
import numpy as np
import tensorflow as tf
import statistics

from attacks import PYY
from models import cnn_1, cnn_2
from datasets import load_mnist, load_cifar10
from tools import evaluate, predict, pseudorandom_target, exclude_miss
from tools import print_out, print_sample


def log(args, path_out, success_query, q, X_data):
    with open(path_out, mode='w') as f:
        for n in success_query:
            f.write(str(n)+'\n')
        
        f.write('-----------------------------\n')
        f.write('all_sample = '+str(len(X_data))+'\n')
        f.write('success_sample = '+str(len(success_query))+'\n\n')

        f.write(str(args.dataset)+'_'+str(args.attack)+' : eps='+str(args.epsilon)+' <'+str('targeted' if args.targeted else 'untargeted')+'>\n')
        f.write('mean:'+str(statistics.mean(success_query))+'\n')
        f.write('median:'+str(statistics.median(success_query))+'\n\n')

        f.write('query\n')
        for cnt in range(1,args.img_size+1):
            f.write('{0:<6}  {1:<6}'.format(cnt*cnt*args.img_chan, q[cnt])+'\n')
        f.write('failed  '+str(q[0])+'\n')


def make_adv(args, sess, env, X_data, y_data):

    print('\nMaking adversarials')

    X_adv = np.empty_like(X_data)
    success_query = []
    q = np.zeros((args.img_size+1), dtype=np.int32)

    for idx, x in enumerate(X_data):
        print(' batch {0}/{1}'.format(idx+1, len(X_data)), end='\r')
        x = np.stack([x])

        ## set target
        if args.targeted:
            target = np.argmax(y_data[idx])
            target = pseudorandom_target(idx, args.n_classes, target)  # int
        else:
            target = np.argmax(y_data[idx])  # int
        
        ## attack
        xadv, num_queries, split, success = eval(args.attack)(args, sess, env, x, target)
        
        ## log
        if success:
            X_adv[idx] = xadv
            q[split] += 1
            success_query.append(num_queries)
        else:
            X_adv[idx] = x
            q[0] += 1
    
    if args.querydata:
        os.makedirs('querydata/'+str(args.dataset)+'_'+str(args.epsilon), exist_ok=True)
        name_out = str(args.dataset)+'_'+str(args.attack)+'_'+str('targeted' if args.targeted else 'untargeted')+'_'+str(args.epsilon)
        path_out = './querydata/'+str(args.dataset)+'_'+str(args.epsilon)+'/'+name_out+'.txt'
        log(args, path_out, success_query, q, X_data)

    return X_adv


def main(args):

    print('\nPreparing {} data'.format(args.dataset))
    if args.dataset == 'mnist':
        (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_mnist()
    elif args.dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_cifar10()
    
    print('\nConstruction graph')
    if args.model == 'cnn_1':
        env = cnn_1(args)
    elif args.model == 'cnn_2':
        env = cnn_2(args)

    print('\nInitializing graph')
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    print('\nLoading saved model')
    name = '{0}_{1}'.format(args.model, args.dataset)
    env.saver.restore(sess, 'models/{0}/{1}'.format(name,name))

    #print('\nEvaluating on clean data')
    #evaluate(sess, env, X_test, y_test)

    print('\nExcluding misclassification samples')
    # mnist 1000 samples -> 0:1010
    # cifar10 1000 samples -> 0:1226
    (X_test, y_test) = exclude_miss(sess, env, X_test, y_test, args.start, args.stop)
    evaluate(sess, env, X_test, y_test)

    print('\nGenerating adversarial data')
    X_adv = make_adv(args, sess, env, X_test, y_test)

    print('\nEvaluating on adversarial data')
    evaluate(sess, env, X_adv, y_test)

    print('\nResults')
    #print_sample(args, sess, env, X_adv, X_test, y_test)
    #print_out(args, sess, env, X_adv, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack', choices=['PYY'], default='PYY')
    parser.add_argument('-d', '--dataset', choices=['mnist', 'cifar10'], default='mnist')
    parser.add_argument('-e', '--epsilon', type=float, default=0.1)
    parser.add_argument('-m', '--model', choices=['cnn_1', 'cnn_2'], default='cnn_1')
    parser.add_argument('-q', '--querydata', action='store_true')
    parser.add_argument('-t', '--targeted', action='store_true')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=1000)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        args.img_size = 28
        args.img_chan = 1
        args.n_classes = 10
    elif args.dataset == 'cifar10':
        args.img_size = 32
        args.img_chan = 3
        args.n_classes = 10
    
    main(args)

