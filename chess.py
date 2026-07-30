import requests
import chess
# requests is a library which sends requests to server , it handles all the socket connection and stufff
# you could code a part of this for your util on your own
user= "Rampranav0511"
oldgames=requests.get(f"https://api.chess.com/pub/player{user}/games/archives") # this is basically a url
# get function sends a http get request to that url requesting for data , in this case - archive games

data=oldgames.json()
"""oldgames is a response object returned by the get function
this contains header , metadata and some content , including status_code,  ok check,url and all of that""" 
# json is another function we call to parse the object file into a json format completely ( basically dictionary here)
l=[]
urls=data["archives"] # archives key holds a collection of links (url's) to my one month games (jan,feb,march.....)
for url in urls:# accessing one particular month every loop
    games=requests.get(url)
    gamesdata=games.json() # contains jan,feb,march , dictionaries
    
    for game in gamesdata["games"]:
        l.append(game["pgn"]) # that particular month's every game pgn is appended into the l list

# rn l is a flat list which contains all game pgn , month by month, linearly
#for ele in l:

    #pgobject=chess.game.read_game(ele) # this function takes in only a file as an input , so we need to write all pgn's into a file
with open("pgn_games","w") as f:
    for ele in l:
        f.write(ele+"/n/n") # blank line after each game
# f.close() , we used with open here so it does it automatically


