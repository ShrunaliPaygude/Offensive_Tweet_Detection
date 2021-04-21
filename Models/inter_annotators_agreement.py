import pandas as pd
from sklearn.metrics import cohen_kappa_score

def main():
    # D:\Files\Courses\Capstone Project\Twitter_API\twitter_search-master\Annotated Data\Amit.xlsx
    files = ['Amit.xlsx', 'Chinmay.xlsx', 'Nikita.xlsx', 'Sayli.xlsx', 'Karan.xlsx', 'Sarthak.xlsx']
    prefix = "Annotated Data//"
    # Amit and Chinmay are a pair
    # Nikita and Sayli are a pair
    # Karan and Sarthak are a pair
    file_contents = []
    for file in files:
        file_content = pd.read_excel(prefix+file)
        file_contents.append(file_content)
    agreements = []

    for annotator in range(0, len(file_contents), 2):
        agreement = cohen_kappa_score(list(file_contents[annotator]['Class'][:100]), list(file_contents[annotator+1]['Class'][:100]))
        agreements.append(agreement)
        print(agreement)
    print("Average agreement: ", sum(agreements) / len(agreements))

def main2():
    files = ['Amit.xlsx', 'Chinmay.xlsx', 'Nikita.xlsx', 'Sayli.xlsx', 'Karan.xlsx', 'Sarthak.xlsx']
    prefix = "Annotated Data//"
    # Amit and Chinmay are a pair
    # Nikita and Sayli are a pair
    # Karan and Sarthak are a pair
    # D:\Files\Courses\Capstone Project\Twitter_API\twitter_search-master\#मराठी
    file_contents = pd.DataFrame()
    remove_duplicate = 0
    for file in files:
        file_content = pd.read_excel(prefix + file)
        if remove_duplicate % 2 == 0:
            file_contents = file_contents.append(file_content)
        else:
            file_contents = file_contents.append(file_content[100:])
        remove_duplicate += 1
        # file_contents.append(file_content)
    # print(len(file_contents))
    file_contents = file_contents[file_contents.Class != 'invalid']
    # file_contents.to_excel(r'Final_Data_Set_For_Models.xlsx', index=False)
    # print(file_contents['Class'].unique())
    print(len(file_contents[file_contents['Class'] == 'invalid']))
    print(len(file_contents[file_contents['Class'] == 'offensive']))
    print(len(file_contents[file_contents['Class'] == 'not offensive']))


if __name__ == '__main__':
    # main()
    # main2()
    pass

