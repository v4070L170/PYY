import numpy as np
import tensorflow as tf


__all__ = [
    'PYY'
]


## xのラベルとtargetに対する確率を返す
def pred(sess, env, x, target):
    ybar = sess.run(env.ybar, feed_dict={env.x: x})
    ybar = np.squeeze(ybar)
    y = np.argmax(ybar)
    proba = ybar[target]
    return y, proba


## 正負を反転
def flip_noise(noise, block):
    noise_new = np.copy(noise)
    noise_new[0, block[0]:block[1], block[2]:block[3], block[4]] *= -1
    return noise_new


##
def PYY(args, sess, env, x, target):
    
    num_queries = 0

    ## make eps all noise
    noise = np.zeros((args.img_size,args.img_size,args.img_chan), dtype=float)
    noise[::] = args.epsilon
    noise = np.stack([noise])

    for split in range(1,args.img_size+1):
        #xadv = np.copy(x)  

        ## 画像の縦(横)を分割数splitで等分
        # 例 : 28を3等分 -> interval[0]=10, interval[1]=9, interval[2]=9
        interval = np.zeros((split), dtype=int)
        for i in range(args.img_size):
            j = interval[i%split]
            interval[i%split] = j + 1

        ## split * split * img_chan で分割しノイズ付加
        for c in range(args.img_chan):
            h_start = 0
            h_end = 0
            for height in range(split):
                h_start = h_end
                h_end = h_end + interval[height]
                w_start = 0
                w_end = 0
                for width in range(split):
                    w_start = w_end
                    w_end = w_end + interval[width]
                    
                    ## クエリ
                    xadv = x + noise
                    # クリッピング
                    xadv = np.clip(xadv, 0., 1.)
                    # prob = 確信度
                    # yadv = モデルのxadvに対する推測クラス
                    yadv, prob = pred(sess, env, xadv, target)
                    num_queries += 1

                    ## 攻撃が成功していた場合はreturn
                    if args.targeted:
                        if yadv == target:
                            return xadv, num_queries, split, True
                    else:
                        if yadv != target:
                            return xadv, num_queries, split, True
                    
                    ## flip noise
                    block = [h_start, h_end, w_start, w_end, c]
                    noise_f = flip_noise(noise, block)
                    
                    ## クエリ
                    xadv_f = x + noise_f
                    # クリッピング
                    xadv_f = np.clip(xadv_f, 0., 1.)
                    # prob_f = 確信度
                    # yadv_f = モデルのxadv_fに対する推測クラス
                    yadv_f, prob_f = pred(sess, env, xadv_f, target)
                    num_queries += 1

                    ## 攻撃が成功していた場合はreturn
                    if args.targeted:
                        if yadv_f == target:
                            return xadv_f, num_queries, split, True
                    else:
                        if yadv_f != target:
                            return xadv_f, num_queries, split, True

                    ## 確信度の比較 -> xadv更新
                    # targeted攻撃のとき
                    if args.targeted:
                        # 確信度が上昇
                        if prob < prob_f:
                            noise = noise_f
                        else:
                            noise = noise
                    # untargeted攻撃のとき
                    else:
                        # 確信度が下降
                        if prob >= prob_f:
                            noise = noise_f
                        else:
                            noise = noise
    
    return x, num_queries, split, False
