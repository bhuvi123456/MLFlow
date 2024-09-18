import argparse
#using this we can provide arguments in the cmd itself  
if __name__  == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--name","-n",default= "Bhuvan",type = str)
    args.add_argument("--age","-a",default=21.0,type = float)
    parse_args = args.parse_args()
    print(parse_args.name,parse_args.age)
#in cmd i can do like this python test.py --name "Bhuvan" --age 21
#Bhuvan 21.0