import random

msg = "Testovací text pro úlohy do předmětu Aplikovaná Kryptografie"

secret_key = [2, 5, 10, 15] 

def encrypt_hus_iter(plaintext, secret_key):
    plaintext = plaintext.lower().replace(" ", "")
    cyphertext = ""
    changed = ""
    for i in range(len(plaintext)):
        if plaintext[i] in ["a", "á"]:
            cyphertext += "b"
            changed += "1"
        elif plaintext[i] in ["e", "é", "ě"]:
            cyphertext += "f"
            changed += "1"
        elif plaintext[i] in ["i", "í"]:
            cyphertext += "j"
            changed += "1"
        elif plaintext[i] in ["o", "ó"]:
            cyphertext += "p"
            changed += "1"
        elif plaintext[i] in ["u", "ú", "ů"]:
            cyphertext += "v"
            changed += "1"
        elif plaintext[i] in ["y", "ý"]:
            cyphertext += "z"
            changed += "1"
        else:
            changed += "0"
            cyphertext += plaintext[i]
    
    changed_list = list(changed)
    for pos in secret_key:
        if pos < len(changed_list):
            changed_list.insert(pos, str(random.randint(0, 1)))
    
    changed_with_random = ''.join(changed_list)
    return cyphertext + "|" + changed_with_random

def decrypt_hus_iter(cyphertext, secret_key):
    encrypted_text, binary_string = cyphertext.split("|")
    
    binary_list = list(binary_string)
    for pos in sorted(secret_key, reverse=True):
        if pos < len(binary_list):
            binary_list.pop(pos)
    
    cleaned_binary = ''.join(binary_list)
    
    plaintext = ""
    binary_index = 0
    for i in range(len(encrypted_text)):
        char = encrypted_text[i]
        if binary_index < len(cleaned_binary) and cleaned_binary[binary_index] == '1':
            if char == 'b':
                plaintext += 'a'
            elif char == 'f':
                plaintext += 'e'
            elif char == 'j':
                plaintext += 'i'
            elif char == 'p':
                plaintext += 'o'
            elif char == 'v':
                plaintext += 'u'
            elif char == 'z':
                plaintext += 'y'
            else:
                plaintext += char
        else:
            plaintext += char
        binary_index += 1  
    
    return plaintext

encrypted_msg = encrypt_hus_iter(msg, secret_key)
print("Encrypted:", encrypted_msg)

decrypted_msg = decrypt_hus_iter(encrypted_msg, secret_key)
print("Decrypted:", decrypted_msg)