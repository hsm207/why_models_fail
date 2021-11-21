import click
import tensorflow as tf
import tensorflow_datasets as tfds

# bsed on https://www.tensorflow.org/datasets/keras_example

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def report_dataset(ds_info):
    print(f'Training set has {ds_info.splits["train"].num_shards} shard(s)')
    print(f'Training set has {ds_info.splits["train"].num_examples:,} examples')
    print(f'Test set has {ds_info.splits["test"].num_examples:,} examples')


def report_metrics(hist):
    print(f"Loss at each epoch")
    for i, loss in enumerate(hist.history["loss"], 1):
        print(f"{i}:\t{loss}")


def evaluate_model(model, ds):
    test_loss, test_acc = model.evaluate(ds, verbose=0)
    print(f"Test loss:\t{test_loss}")
    print(f"Test accuracy:\t{test_acc}")


def build_pipelines(shuffle_data, num_examples):
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=shuffle_data,
        as_supervised=True,
        with_info=True,
    )

    report_dataset(ds_info)

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE).take(
        num_examples
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test


def build_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


@click.command()
@click.option("--seed", default=None, help="Value of the global seed to use", type=int)
@click.option("--shuffle_data", default=True, help="Whether to shuffle the data")
@click.option(
    "--num_examples",
    default=60000,
    help="Number of examples in the training set to actually use for training",
)
def main(seed, shuffle_data, num_examples):

    tf.random.set_seed(seed)

    ds_train, ds_test = build_pipelines(shuffle_data, num_examples)

    if not shuffle_data:
        print("Training data will not be shuffled")

    if seed:
        print(f"Setting global random seed to {seed}")
    
    model = build_model()

    hist = model.fit(ds_train, epochs=6, validation_data=ds_test, verbose=0)

    report_metrics(hist)
    evaluate_model(model, ds_test)


if __name__ == "__main__":
    main()
