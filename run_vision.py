# run_vision.py — Experiment 1 (Vision-only baseline: healthy vs unhealthy)
import itertools, json, os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

SEED=13; IMG_SIZE=(224,224); BATCH=32; EPOCHS=12
np.random.seed(SEED); tf.random.set_seed(SEED)

H_DIR = Path("data/images/healthy")
U_DIR = Path("data/images/unhealthy")
OUT   = Path("results/vision"); OUT.mkdir(parents=True, exist_ok=True)

def make_df():
    import pandas as pd
    rows=[]
    for lblname,lbl,root in [("healthy",0,H_DIR),("unhealthy",1,U_DIR)]:
        for p in itertools.chain(root.glob("*.jpg"), root.glob("*.jpeg"), root.glob("*.png"), root.glob("*.JPG"), root.glob("*.PNG")):
            rows.append({"path":str(p), "label":lbl})
    df = pd.DataFrame(rows)
    if df.empty: raise SystemExit("No images found. Put files in data/images/healthy and /unhealthy")
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def ds_from_df(df, training):
    paths = df["path"].values; labels = df["label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    def load(path, y):
        img=tf.io.read_file(path); img=tf.image.decode_image(img, channels=3, expand_animations=False)
        img=tf.image.resize(img, IMG_SIZE); img=tf.cast(img, tf.float32)/255.0
        return img, tf.one_hot(y,2)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        aug = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x,y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(2048, seed=SEED)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def build_model():
    base = keras.applications.MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights="imagenet")
    base.trainable=False
    inp=keras.Input(shape=IMG_SIZE+(3,))
    x=base(inp, training=False)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(0.25)(x)
    out=layers.Dense(2, activation="softmax")(x)
    model=keras.Model(inp,out)
    model.compile(optimizer=keras.optimizers.Adam(3e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def plot_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix"); ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,cm[i,j],ha="center",va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def plot_roc(y_true, y_score, path):
    fpr,tpr,_=roc_curve(y_true, y_score)
    plt.figure(figsize=(4,4)); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.tight_layout(); plt.savefig(path); plt.close()

if __name__=="__main__":
    import pandas as pd
    # Collect files
    df = make_df()
    print(f"Total images: {len(df)} (healthy={sum(df.label==0)}, unhealthy={sum(df.label==1)})")

    # 70/15/15 split
    n=len(df); n_train=int(0.7*n); n_val=int(0.15*n)
    train_df=df.iloc[:n_train]; val_df=df.iloc[n_train:n_train+n_val]; test_df=df.iloc[n_train+n_val:]
    print(f"Split sizes → train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Datasets
    ds_tr = ds_from_df(train_df, True)
    ds_va = ds_from_df(val_df,   False)
    ds_te = ds_from_df(test_df,  False)

    # Class weights (helps with imbalance)
    cc = np.bincount(train_df["label"].values)
    class_weight = {i: float(cc.sum())/(2.0*cc[i]) for i in range(2)}
    print("Class weights:", class_weight)

    # Train
    model = build_model()
    ckpt = keras.callbacks.ModelCheckpoint(OUT/"best.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
    es   = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    hist = model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS, callbacks=[ckpt, es],
                     class_weight=class_weight, verbose=2)

    # Evaluate on TEST
    y_true = test_df["label"].values
    y_prob = model.predict(ds_te, verbose=0)
    y_pred = y_prob.argmax(axis=1)
    try: auc = float(roc_auc_score(y_true, y_prob[:,1]))
    except: auc = float("nan")

    rep = classification_report(y_true, y_pred, output_dict=True, digits=4)
    cm  = confusion_matrix(y_true, y_pred)

    # Save artifacts
    json.dump({"auc":auc, "report":rep, "class_weight":class_weight}, open(OUT/"classification_report.json","w"), indent=2)
    plot_confusion(cm, ["healthy","unhealthy"], OUT/"confusion_matrix.png")
    plot_roc(y_true, y_prob[:,1], OUT/"roc.png")

    model.save(OUT/"model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    open(OUT/"model.tflite","wb").write(converter.convert())

    print("\n[Vision] Done. See results in", OUT)
