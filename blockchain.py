import json
import time
import hashlib


class Block:
    def __init__(self, data, previous_hash):
        self.data = data
        self.previousHash = previous_hash
        self.timestamp = time.time()
        self.nonce = 0
        self.hash = self.calculate_hash()
        


    def __str__(self): 
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    
    def calculate_hash(self):
        text = "" + self.data + self.previousHash + str(self.timestamp) + str(self.nonce)
        hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return hash
    
    def mine_block(self, difficulty):
        test = True
        while test:
            newHash = self.calculate_hash()
            if newHash[:difficulty] != "0"*difficulty:
                self.nonce += 1
            else:
                self.hash = newHash
                test = False

    def edit_block(self, newData = 0, newHash = 0, newNonce = 0, newPreviousHash = 0, newTimestamp = 0):
        if newData != 0:
            self.data = newData
        if newHash != 0:
            self.hash = newHash
        if newNonce != 0:
            self.Nonce = newNonce
        if newPreviousHash != 0:
            self.previousHash = newPreviousHash
        if newTimestamp != 0:
            self.timestamp = newTimestamp
        
      
        

def is_chain_valid(blockchain):
    hash = "0";
    for block in blockchain:
        if block.previousHash != hash:
            return False
        text = "" + block.data + block.previousHash + str(block.timestamp) + str(block.nonce)
        hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if hash != block.hash:
            return False
        
    return True
    
def print_blockchain(blockchain):
    print("\nBlockchain:")
    for block in blockchain:
        print(block)
    
    
difficulty = 4
blockchain = []

blockchain.append(Block("Ahoj, ja jsem prvni blok.", "0"))
print("Tezim blok 1...")
blockchain[0].mine_block(difficulty)
print("Blok vytezen! " + blockchain[0].hash)

blockchain.append(Block("Ja jsem druhy blok.", blockchain[0].hash))
print("Tezim blok 2...")
blockchain[1].mine_block(difficulty)
print("Blok vytezen! " + blockchain[1].hash)

blockchain.append(Block("Ja jsem treti blok.", blockchain[1].hash))
print("Tezim blok 3...")
blockchain[2].mine_block(difficulty)
print("Blok vytezen! " + blockchain[2].hash)


print_blockchain(blockchain)
print("Blockchain je platny: " + str(is_chain_valid(blockchain)))

blockchain[1].edit_block("test")
print("Blockchain je platny: " + str(is_chain_valid(blockchain)))

print("---------------------------------------------------------------------------")