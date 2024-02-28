import streamlit as st
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def main():
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    class_names = {0: "T-shirt/top",
                   1: "Trouser",
                   2: "Pullover",
                   3: "Dress",
                   4: "Coat",
                   5: "Sandal",
                   6: "Shirt",
                   7: "Sneaker",
                   8: "Bag",
                   9: "Ankle boot"}
    df = pd.DataFrame(class_names.items(), columns=["Label", "Category"])
    model = tf.keras.models.load_model('mnist_model.h5', compile=False)
    menu = ["Data", "Model"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Data":
        st.title("Multi Class Classification of Fashion MNIST")
        st.subheader("Dataset")
        st.write('''The MNIST dataset consists of 70,000 28x28 black-and-white images in 10 classes (one for each digits), with 7,000
images per class. There are 60,000 training images and 10,000 test images.
The dataset has the following labels''')
        st.dataframe(df, use_container_width=True,hide_index=True)
        st.subheader("Data Samples")
        st.write('Some of the images in the dataset can be generated here')
        imgs = []
        label = []
        for i in range(6):
            index_of_choice = i
            imgs.append(train_data[index_of_choice])
            label.append(class_names[train_labels[index_of_choice]])
        if st.button('Generate Random Sample', type="primary"):
            for i in range(6):
                index_of_choice = random.randint(0, len(train_data))
                imgs[i] = (train_data[index_of_choice])
                label[i] = (class_names[train_labels[index_of_choice]])
            st.image(imgs, width=200, caption=label)
        else:
            st.image(imgs, width=200, caption=label)


        st.subheader("Predictions")
        if st.button('Predict', type="primary"):
            test_data_norm = test_data/255
            i = random.randint(0, len(test_data_norm))

            target_image = test_data_norm[i]
            pred_probs = model.predict(target_image.reshape(1, 28, 28))
            pred_label = class_names[pred_probs.argmax()]
            true_label = class_names[test_labels[i]]

            st.image(target_image, width=200)
            if pred_label == true_label:
                st.success("Prediction: {}".format(pred_label))
                st.write("**Prediction Probability**: {:2.0f}%".format(100*tf.reduce_max(pred_probs)))
                st.write("**True Label**: {}".format(true_label))
            else:
                st.error("Prediction: {}".format(pred_label))
                st.write("**Prediction Probability**: {:2.0f}%".format(100 * tf.reduce_max(pred_probs)))
                st.write("**True Label**: {}".format(true_label))

    elif choice == "Model":
        st.title("Multi Class Classification of Fashion MNIST")
        st.subheader("Model Architecture")
        st.write("The model architecture is as follows:")
        st.image("model.png")
        st.subheader("Confusion Matrix")
        st.write("The confusion matrix is as follows:")
        st.image("confusion_m.png")
        st.subheader("Epoch vs Loss")
        st.image("output.png")


if __name__ == "__main__":
    main()
