import discord
import random
import os
import requests
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np  

from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='.', intents=intents)


def get_class(image_path, model_path, labels_path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(model_path, compile=False)

    # Load the labels
    class_names = open(labels_path, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)



@bot.command()
async def hello(ctx):
    await ctx.send(f'Привет! Я бот {bot.user}!')


@bot.command()
async def add(ctx, left: int, right: int):
    """Adds two numbers together."""
    await ctx.send(left + right) 

@bot.command()
async def roll(ctx, dice: str):
    """Rolls a dice in NdN format."""
    try:
        rolls, limit = map(int, dice.split('d'))
    except Exception:
        await ctx.send('Format has to be in NdN!')
        return

    result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
    await ctx.send(result)


@bot.command()
async def repeat(ctx, times: int, content='repeating...'):
    """Repeats a message multiple times."""
    for i in range(times):
        await ctx.send(content)

@bot.command()
async def triple(ctx, z: int, x: int, c: int):
    await ctx.send(z + x + c)

@bot.command()
async def ping(ctx):
    await ctx.send(f'Pong! {round (bot.latency * 1000)}ms ')

@bot.command()
async def info(ctx):
    await ctx.send(ctx.message.attachments)

@bot.command()
async def quote(ctx):
    responses = open('quetes.txt').read().splitlines()
    random.seed(a=None)
    response = random.choice(responses)
    await ctx.send(response)

@bot.command()
async def aimg(ctx):
    for i in ctx.message.attachments:
        number = random.randint(0, 1000)
        await i.save(f'uploaded_image{number}.png')
        result = get_class(f'uploaded_image{number}.png', 'keras_model.h5', 'labels.txt')
        await ctx.send(result)

@bot.event
async def on_command_error(ctx, error):
    await ctx.send(f'Error ({error})')




bot.run("")


