{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b7c35449-9f9c-4b11-b7fb-622b2e4dd1f6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: roboflow in c:\\users\\azizb\\anaconda3\\lib\\site-packages (1.1.49)\n",
      "Requirement already satisfied: certifi in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (2024.6.2)\n",
      "Requirement already satisfied: idna==3.7 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (3.7)\n",
      "Requirement already satisfied: cycler in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (0.11.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (1.4.4)\n",
      "Requirement already satisfied: matplotlib in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (3.8.4)\n",
      "Requirement already satisfied: numpy>=1.18.5 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (1.26.4)\n",
      "Requirement already satisfied: opencv-python-headless==4.10.0.84 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (4.10.0.84)\n",
      "Requirement already satisfied: Pillow>=7.1.2 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (10.3.0)\n",
      "Requirement already satisfied: python-dateutil in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (2.9.0.post0)\n",
      "Requirement already satisfied: python-dotenv in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (0.21.0)\n",
      "Requirement already satisfied: requests in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (2.32.2)\n",
      "Requirement already satisfied: six in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (1.16.0)\n",
      "Requirement already satisfied: urllib3>=1.26.6 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (2.2.2)\n",
      "Requirement already satisfied: tqdm>=4.41.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (4.66.4)\n",
      "Requirement already satisfied: PyYAML>=5.3.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (6.0.1)\n",
      "Requirement already satisfied: requests-toolbelt in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (1.0.0)\n",
      "Requirement already satisfied: filetype in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from roboflow) (1.2.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tqdm>=4.41.0->roboflow) (0.4.6)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from matplotlib->roboflow) (1.2.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from matplotlib->roboflow) (4.51.0)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from matplotlib->roboflow) (23.2)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from matplotlib->roboflow) (3.0.9)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from requests->roboflow) (2.0.4)\n"
     ]
    }
   ],
   "source": [
    "!pip install roboflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6bb49b9c-6f85-470f-9239-526eae2b1c8f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tensorflow in c:\\users\\azizb\\anaconda3\\lib\\site-packages (2.18.0)\n",
      "Requirement already satisfied: tensorflow-intel==2.18.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow) (2.18.0)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.1.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=24.3.25 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (24.3.25)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.6.0)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (18.1.1)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.4.0)\n",
      "Requirement already satisfied: packaging in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (23.2)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.20.3)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.32.2)\n",
      "Requirement already satisfied: setuptools in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (69.5.1)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.5.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (4.11.0)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.14.1)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.67.1)\n",
      "Requirement already satisfied: tensorboard<2.19,>=2.18 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (2.18.0)\n",
      "Requirement already satisfied: keras>=3.5.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.6.0)\n",
      "Requirement already satisfied: numpy<2.1.0,>=1.26.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (1.26.4)\n",
      "Requirement already satisfied: h5py>=3.11.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (3.11.0)\n",
      "Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow) (0.4.1)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.18.0->tensorflow) (0.43.0)\n",
      "Requirement already satisfied: rich in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (13.3.5)\n",
      "Requirement already satisfied: namex in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (0.0.8)\n",
      "Requirement already satisfied: optree in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (0.13.1)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow) (2024.6.2)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (0.7.2)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (3.0.3)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow) (2.1.3)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from rich->keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from rich->keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\azizb\\anaconda3\\lib\\site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich->keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow) (0.1.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install tensorflow"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "105b958e-cebb-49bf-a145-3eb97b1eaf45",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Files extracted to: C:/Users/azizb/Documents/MyExtractedData\n"
     ]
    }
   ],
   "source": [
    "import zipfile\n",
    "import os\n",
    "\n",
    "# Define the path to the ZIP file and the output directory\n",
    "zip_file_path = 'C:/Users/azizb/Downloads/getChess.v3i.tensorflow.zip'  # Corrected user folder name\n",
    "output_directory = 'C:/Users/azizb/Documents/MyExtractedData'  # Ensure this is a folder where you have full write access\n",
    "  # Ensure this is the correct path\n",
    "\n",
    "# Create the output directory if it doesn't exist\n",
    "os.makedirs(output_directory, exist_ok=True)\n",
    "\n",
    "# Extract the ZIP file\n",
    "with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:\n",
    "    zip_ref.extractall(output_directory)\n",
    "\n",
    "print(f\"Files extracted to: {output_directory}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "08a8b2a7-8854-4c33-9f54-7f78561ebc27",
   "metadata": {},
   "outputs": [],
   "source": [
    " from tensorflow.keras.preprocessing.image import ImageDataGenerator"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "5e089669-9730-418f-ae1b-31164dec07f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "2e6bbd48-bbfe-4005-bdd9-ff11f4579636",
   "metadata": {},
   "outputs": [],
   "source": [
    " dataset_path = \"C:/Users/azizb/Downloads/getChess.v3i.tensorflow\"  # Replace with the actual path to your \n",
    " train_dir = os.path.join(dataset_path, 'train')\n",
    " valid_dir = os.path.join(dataset_path, 'valid')\n",
    " test_dir = os.path.join(dataset_path, 'test')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ef69193f-36e6-49f5-8729-5c4578742b5d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 18070 images belonging to 3 classes.\n",
      "Found 18070 images belonging to 3 classes.\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "\n",
    "# Define your directories for training and validation data\n",
    "train_dir = 'C:/Users/azizb/Documents/MyExtractedData'  # Replace with the actual path to your training directory\n",
    "valid_dir = 'C:/Users/azizb/Documents/MyExtractedData'  # Replace with the actual path to your validation directory\n",
    "\n",
    "# Create an instance of ImageDataGenerator for training and validation\n",
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1.0/255,             # Rescale pixel values from 0-255 to 0-1\n",
    "    rotation_range=20,           # Apply random rotations\n",
    "    width_shift_range=0.2,       # Apply horizontal shifts\n",
    "    height_shift_range=0.2,      # Apply vertical shifts\n",
    "    shear_range=0.2,             # Shearing\n",
    "    zoom_range=0.2,              # Zooming\n",
    "    horizontal_flip=True,        # Horizontal flip\n",
    "    fill_mode='nearest'          # Fill in missing pixels\n",
    ")\n",
    "\n",
    "valid_datagen = ImageDataGenerator(rescale=1.0/255)  # Only rescale for validation data\n",
    "\n",
    "# Load images from directories\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    train_dir,\n",
    "    target_size=(224, 224),       # Resize images to 224x224 for the model \n",
    "    batch_size=32,\n",
    "    class_mode='categorical'      # Use 'categorical' for multiclass classification\n",
    ")\n",
    "\n",
    "valid_generator = valid_datagen.flow_from_directory(\n",
    "    valid_dir,\n",
    "    target_size=(224, 224),\n",
    "    batch_size=32,\n",
    "    class_mode='categorical'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "57b3de96-39b9-4098-8da5-f0535cd8c2f5",
   "metadata": {},
   "outputs": [],
   "source": [
    " from tensorflow.keras.applications import MobileNetV2\n",
    " from tensorflow.keras.layers import Dense, GlobalAveragePooling2D\n",
    " from tensorflow.keras.models import Model\n",
    " # Load a pre-trained model and customize it\n",
    " base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, \n",
    "weights='imagenet')\n",
    " base_model.trainable = False  # Freeze base model layers\n",
    " # Add custom layers on top\n",
    " x = base_model.output\n",
    " x = GlobalAveragePooling2D()(x)\n",
    " x = Dense(128, activation='relu')(x)\n",
    " predictions = Dense(train_generator.num_classes, activation='softmax')(x)\n",
    " model = Model(inputs=base_model.input, outputs=predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cd5df194-04f9-4de0-9dae-a107874a65ae",
   "metadata": {},
   "outputs": [],
   "source": [
    " model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=\n",
    " ['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "5816352c-79ef-4144-9efd-2b6ccb0c5184",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.utils import Sequence\n",
    "\n",
    "class CustomDataset(Sequence):\n",
    "    def __init__(self, *args, **kwargs):\n",
    "        # Ensure you call the parent constructor\n",
    "        super().__init__(**kwargs)\n",
    "        # Your custom dataset initialization code here\n",
    "        self.data = args\n",
    "\n",
    "    def __len__(self):\n",
    "        # Define how many batches per epoch\n",
    "        return len(self.data) // self.batch_size\n",
    "\n",
    "    def __getitem__(self, index):\n",
    "        # Return a batch of data\n",
    "        return self.data[index]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "b5275d9d-ad33-4d39-ac3f-c61fe3a24e1b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\azizb\\anaconda3\\Lib\\site-packages\\keras\\src\\trainers\\data_adapters\\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
      "  self._warn_if_super_not_called()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 951ms/step - accuracy: 0.8590 - loss: 0.4556"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\azizb\\anaconda3\\Lib\\site-packages\\keras\\src\\trainers\\data_adapters\\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
      "  self._warn_if_super_not_called()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m731s\u001b[0m 1s/step - accuracy: 0.8590 - loss: 0.4555 - val_accuracy: 0.8675 - val_loss: 0.4020\n",
      "Epoch 2/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m589s\u001b[0m 1s/step - accuracy: 0.8694 - loss: 0.3752 - val_accuracy: 0.8675 - val_loss: 0.3691\n",
      "Epoch 3/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m568s\u001b[0m 1s/step - accuracy: 0.8691 - loss: 0.3665 - val_accuracy: 0.8675 - val_loss: 0.3767\n",
      "Epoch 4/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m591s\u001b[0m 1s/step - accuracy: 0.8652 - loss: 0.3718 - val_accuracy: 0.8675 - val_loss: 0.3768\n",
      "Epoch 5/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m511s\u001b[0m 904ms/step - accuracy: 0.8679 - loss: 0.3651 - val_accuracy: 0.8675 - val_loss: 0.3697\n",
      "Epoch 6/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m512s\u001b[0m 907ms/step - accuracy: 0.8672 - loss: 0.3607 - val_accuracy: 0.8675 - val_loss: 0.3753\n",
      "Epoch 7/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m559s\u001b[0m 990ms/step - accuracy: 0.8704 - loss: 0.3576 - val_accuracy: 0.8675 - val_loss: 0.3686\n",
      "Epoch 8/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m561s\u001b[0m 993ms/step - accuracy: 0.8718 - loss: 0.3537 - val_accuracy: 0.8675 - val_loss: 0.3646\n",
      "Epoch 9/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m540s\u001b[0m 956ms/step - accuracy: 0.8637 - loss: 0.3670 - val_accuracy: 0.8675 - val_loss: 0.3630\n",
      "Epoch 10/10\n",
      "\u001b[1m565/565\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m526s\u001b[0m 931ms/step - accuracy: 0.8706 - loss: 0.3529 - val_accuracy: 0.8675 - val_loss: 0.3582\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(\n",
    "    train_generator,        # Training data generator\n",
    "    epochs=10,              # Number of epochs\n",
    "    validation_data=valid_generator  # Validation data generator\n",
    ")\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "1c52256a-ff7a-402c-a6e2-3e4f2593815f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x_batch shape: (32, 224, 224, 3), y_batch shape: (32, 3)\n"
     ]
    }
   ],
   "source": [
    " # Check if the generator is working properly\n",
    "x_batch, y_batch = next(train_generator)\n",
    "print(f\"x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7ca1cab1-8951-47d9-8af7-82aa03de5c91",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 0 images belonging to 0 classes.\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "Must provide at least one structure",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[13], line 10\u001b[0m\n\u001b[0;32m      1\u001b[0m test_generator \u001b[38;5;241m=\u001b[39m valid_datagen\u001b[38;5;241m.\u001b[39mflow_from_directory(\n\u001b[0;32m      2\u001b[0m     test_dir,\n\u001b[0;32m      3\u001b[0m     target_size\u001b[38;5;241m=\u001b[39m(\u001b[38;5;241m224\u001b[39m, \u001b[38;5;241m224\u001b[39m),   \u001b[38;5;66;03m# Ensure the test images are resized to 224x224\u001b[39;00m\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m      6\u001b[0m     shuffle\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m            \u001b[38;5;66;03m# Disable shuffling for consistent evaluation\u001b[39;00m\n\u001b[0;32m      7\u001b[0m )\n\u001b[0;32m      9\u001b[0m \u001b[38;5;66;03m# Evaluate the model on the test dataset\u001b[39;00m\n\u001b[1;32m---> 10\u001b[0m test_loss, test_accuracy \u001b[38;5;241m=\u001b[39m model\u001b[38;5;241m.\u001b[39mevaluate(test_generator)\n\u001b[0;32m     12\u001b[0m \u001b[38;5;66;03m# Print the test accuracy\u001b[39;00m\n\u001b[0;32m     13\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mTest Accuracy: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mtest_accuracy\u001b[38;5;250m \u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;250m \u001b[39m\u001b[38;5;241m100\u001b[39m\u001b[38;5;132;01m:\u001b[39;00m\u001b[38;5;124m.2f\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m%\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\keras\\src\\utils\\traceback_utils.py:122\u001b[0m, in \u001b[0;36mfilter_traceback.<locals>.error_handler\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    119\u001b[0m     filtered_tb \u001b[38;5;241m=\u001b[39m _process_traceback_frames(e\u001b[38;5;241m.\u001b[39m__traceback__)\n\u001b[0;32m    120\u001b[0m     \u001b[38;5;66;03m# To get the full stack trace, call:\u001b[39;00m\n\u001b[0;32m    121\u001b[0m     \u001b[38;5;66;03m# `keras.config.disable_traceback_filtering()`\u001b[39;00m\n\u001b[1;32m--> 122\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m e\u001b[38;5;241m.\u001b[39mwith_traceback(filtered_tb) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m    123\u001b[0m \u001b[38;5;28;01mfinally\u001b[39;00m:\n\u001b[0;32m    124\u001b[0m     \u001b[38;5;28;01mdel\u001b[39;00m filtered_tb\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\keras\\src\\tree\\optree_impl.py:76\u001b[0m, in \u001b[0;36mmap_structure\u001b[1;34m(func, *structures)\u001b[0m\n\u001b[0;32m     74\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m`func` must be callable. Received: func=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfunc\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     75\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m structures:\n\u001b[1;32m---> 76\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mMust provide at least one structure\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     77\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m other \u001b[38;5;129;01min\u001b[39;00m structures[\u001b[38;5;241m1\u001b[39m:]:\n\u001b[0;32m     78\u001b[0m     assert_same_structure(structures[\u001b[38;5;241m0\u001b[39m], other, check_types\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n",
      "\u001b[1;31mValueError\u001b[0m: Must provide at least one structure"
     ]
    }
   ],
   "source": [
    "test_generator = valid_datagen.flow_from_directory(\n",
    "    test_dir,\n",
    "    target_size=(224, 224),   # Ensure the test images are resized to 224x224\n",
    "    batch_size=32,           # Use a batch size of 32\n",
    "    class_mode='categorical',# Use 'categorical' for multi-class classification\n",
    "    shuffle=False            # Disable shuffling for consistent evaluation\n",
    ")\n",
    "\n",
    "# Evaluate the model on the test dataset\n",
    "test_loss, test_accuracy = model.evaluate(test_generator)\n",
    "\n",
    "# Print the test accuracy\n",
    "print(f\"Test Accuracy: {test_accuracy * 100:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "5b74713e-cc44-4db8-ba14-44135d84799f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "# Save the entire model as a .h5 file\n",
    "model.save(\"chess_armrobot_model.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "ec6f8c49-b6b2-4b90-8357-7520e89424e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "# Load the entire model\n",
    "model = load_model(\"chess_armrobot_model.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "af85635d-6246-4490-a560-d7483ad68c2e",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Must provide at least one structure",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[18], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# Test the reloaded model\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m test_loss, test_accuracy \u001b[38;5;241m=\u001b[39m model\u001b[38;5;241m.\u001b[39mevaluate(test_generator)\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mTest accuracy: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mtest_accuracy\u001b[38;5;250m \u001b[39m\u001b[38;5;241m*\u001b[39m\u001b[38;5;250m \u001b[39m\u001b[38;5;241m100\u001b[39m\u001b[38;5;132;01m:\u001b[39;00m\u001b[38;5;124m.2f\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m%\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\keras\\src\\utils\\traceback_utils.py:122\u001b[0m, in \u001b[0;36mfilter_traceback.<locals>.error_handler\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    119\u001b[0m     filtered_tb \u001b[38;5;241m=\u001b[39m _process_traceback_frames(e\u001b[38;5;241m.\u001b[39m__traceback__)\n\u001b[0;32m    120\u001b[0m     \u001b[38;5;66;03m# To get the full stack trace, call:\u001b[39;00m\n\u001b[0;32m    121\u001b[0m     \u001b[38;5;66;03m# `keras.config.disable_traceback_filtering()`\u001b[39;00m\n\u001b[1;32m--> 122\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m e\u001b[38;5;241m.\u001b[39mwith_traceback(filtered_tb) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m    123\u001b[0m \u001b[38;5;28;01mfinally\u001b[39;00m:\n\u001b[0;32m    124\u001b[0m     \u001b[38;5;28;01mdel\u001b[39;00m filtered_tb\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\keras\\src\\tree\\optree_impl.py:76\u001b[0m, in \u001b[0;36mmap_structure\u001b[1;34m(func, *structures)\u001b[0m\n\u001b[0;32m     74\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m`func` must be callable. Received: func=\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mfunc\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     75\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m structures:\n\u001b[1;32m---> 76\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mMust provide at least one structure\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     77\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m other \u001b[38;5;129;01min\u001b[39;00m structures[\u001b[38;5;241m1\u001b[39m:]:\n\u001b[0;32m     78\u001b[0m     assert_same_structure(structures[\u001b[38;5;241m0\u001b[39m], other, check_types\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n",
      "\u001b[1;31mValueError\u001b[0m: Must provide at least one structure"
     ]
    }
   ],
   "source": [
    "# Test the reloaded model\n",
    "test_loss, test_accuracy = model.evaluate(test_generator)\n",
    "print(f\"Test accuracy: {test_accuracy * 100:.2f}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "fe259dc6-8d3e-4b30-8720-c100600487f2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    }
   ],
   "source": [
    "# Save in the current directory\n",
    "model.save(\"./chess_armrobot_model.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f73a0809-c95f-4810-9793-3b2342098814",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
