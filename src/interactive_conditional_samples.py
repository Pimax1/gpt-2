#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import time

import model, sample, encoder
from google.cloud import storage

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=27,
    batch_size=27,
    length=20,
    temperature=0.7,
    top_k=30,
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        gcs = storage.Client()
        bucket = gcs.get_bucket('pimax')
        question_file = r"gpt-2/question.json"
        prediction_file = r"gpt-2/predictions.json"
        nb_iter = 3
        total_predictions = nsamples  # ** nb_iter

        while True:

            if not storage.Blob(bucket=bucket, name=question_file).exists(gcs):
                time.sleep(1)
                continue
            raw_text = json.loads(bucket.get_blob(question_file).download_as_string())["question"]
            # raw_text = bucket.get_blob(r'117M/text.txt').download_as_string().decode("utf-8")
            preds_tmp = {}
            for i in range(1, total_predictions + 1):
                preds_tmp["answer_%s" % str(i)] = [raw_text]

            #while we have not fully completed the answers
            while len(preds_tmp["answer_%s" % str(total_predictions)]) != nb_iter + 1:
                next_pred = getnextpred(preds_tmp)
                iter = len(preds_tmp["answer_" + str(next_pred)])
                nb_answers_to_fill = nsamples**(nb_iter-iter)
                generated = 0
                next_sentence = getnextsentence(preds_tmp)
                context_tokens = enc.encode(next_sentence)
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                        print(text)
                        print(clean_sentence(text))
                        preds_tmp = addNewPreds(clean_sentence(text), preds_tmp, 1)
                        #preds_tmp = addNewPreds(clean_sentence(text), preds_tmp, nb_answers_to_fill)
                print("=" * 80)
            predictions = {}
            predictions["answers"] = []
            for i in range(1, total_predictions+1):
                answer = "".join(preds_tmp["answer_%s" % str(i)])  # join answers
                answer = ".".join(list(dict.fromkeys(answer.split("."))))  # remove duplicate sentences
                answer = "!".join(list(dict.fromkeys(answer.split("!"))))  # remove duplicate sentences
                answer = "?".join(list(dict.fromkeys(answer.split("?"))))  # remove duplicate sentences
                answer = answer[:max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"), 0) + 1]  # till last .!?
                predictions["answers"].append(answer)
            bucket.blob(question_file).delete()
            bucket.blob(prediction_file).upload_from_string(json.dumps(predictions, ensure_ascii=False))


def getnextpred(predictions):
    """
    :param predictions:
    :return: indice of next prediction to fill
    """
    nextpred = 1
    maxpreds = len(predictions["answer_1"])
    for i in range(1, len(predictions)+1):
        if maxpreds != len(predictions["answer_%s" % str(i)]):
            nextpred = i
            break
    return nextpred


def addNewPreds(text, predictions, nb_times):
    nextpred = getnextpred(predictions)
    for i in range(nextpred, nextpred + nb_times):
        predictions["answer_%s" % str(i)].append(text)
    return predictions


def getnextsentence(predictions):
    nextpred = getnextpred(predictions)
    return "".join(predictions["answer_%s" % str(nextpred)])

def clean_sentence(text):
    text = text.replace("\n", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("<|endoftext|>", "")

    text = text.replace(" A.", " ").replace(" a.", " ").replace(" a,", " ").replace(" A,", " ").replace(" A:", " ")
    text = text.replace(" A;", " ").replace(" a;", " ").replace("\"", "")
    text = text.replace(" A;", " ").replace(" a;", " ").replace("\"", "")



    # text = text.split("\n")[0]
    # text = text[:max(text.rfind(";"), text.rfind("!"),text.rfind("."), 0)]
    return text

if __name__ == '__main__':
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'D:\onedrive\dev\GCP\thematic-envoy-235713-727f0c0b00de.json'
    os.environ['PYTHONIOENCODING'] = 'UTF-8'
    fire.Fire(interact_model)

