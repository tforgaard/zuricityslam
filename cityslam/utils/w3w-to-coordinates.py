
##//INCLUDE-Api-Client-to-interface-w3w-API-------------------------------////
import what3words as w3w #pip install what3words

##//INCLUDE-Aditionals----------------------------------------------------////
from oauth2client.tools import argparser #pip install oauth2client

def main():
    global w3wApiKey
    w3wApiKey="AWXNVR24" #Input your w3wApiKey
    global thisLocationCO
    thisLocationCO = "47.371667, 8.542222"
    #inputtype = "w3w"
    #input = "///trailer.sung.believer"
    input_w = input("Enter What3Words adress: ")
    thisLocationCO = w3w_to_CO(input_w)

##//METHODES-GEOPOS-------------------------------------------------------////
def w3w_to_CO(input):
    print("w3w_to_CO from: " + str(input))
    geocoder = w3w.Geocoder(w3wApiKey)
    result = geocoder.convert_to_coordinates(input)
    #print("w3w_to_CO to: " + str(result))
    out = str(result['coordinates']['lat']) + ", " + str(result['coordinates']['lng'])
    print("w3w_to_CO   to: " + str(out))
    return out

if __name__ == "__main__":
    main()