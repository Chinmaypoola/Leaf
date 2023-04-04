import discord
from discord.ext import commands
import model

PERMISSIONS_INTEGER = 326417984704
client = commands.Bot(command_prefix = '.', intents = discord.Intents.all())

global chat 

client.remove_command('help')

@client.event
async def on_ready():
  for i in range(3):
    print("\n")
  print(f"we have logged in as {client.user}")
  
@client.command()
async def help(ctx):
  embed = discord.Embed(
    title= "Help for you",
    color= discord.Colour.gold()
  )
  embed.add_field(name= '.help',value='Using this command shows this embedded message')
  embed.add_field(name= '.setup',value='This used for you to setup the channel for me to talk with you')
  embed.add_field(name= '.l',value='This used for you to talk with me :). So you kill time by getting a humanly response')
  await ctx.send(embed=embed)

@client.command()
async def setup(ctx, *, given_name=None):
  list_of_channels = {}
  for channel in ctx.guild.text_channels:
    list_of_channels[str(channel)] = channel.id
  try:
    chat = list_of_channels["leaf"]
    chat = client.get_channel(chat)
    await chat.send('I\'ve already been setup')
  except:
    create_chat = await ctx.guild.create_text_channel("leaf")
    await create_chat.send('heyy you can talk with me in this channel')
  list_of_channels = {}

@client.command()
async def l(ctx , *,msg):
  if (str(ctx.channel) == "leaf"):
    response = model.response(msg)
    print(f"message : {msg}\nresponse : {response}")
    await ctx.send(response)
  else:
    print(f"message was sent in {str(ctx.channel)}")
    await ctx.send("Please text in the channel leaf")

#here you have to use your token which is provided by discord. The following comment is just my way of doing it
"""f = open("C:\\Users\\chinm\\OneDrive\\Desktop\\college work\\Sem 5\\ai\\Project\\token.txt","r")
var = f.read()"""
client.run("Token") #place your token here
