import os
from chromadb import Client, Settings
from clip_embeddings import ClipEmbeddingsfunction
from typing import List

ef = ClipEmbeddingsfunction()
client = Client(
    settings=Settings(is_persistent=True, persist_directory="./clip_chroma")
)
coll = client.get_or_create_collection(name="clip", embedding_function=ef)


def get_docs(dir_path: str) -> List[str]:
    docs = []

    for file in os.listdir(dir_path):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            docs.append(dir_path + "/" + file)

    return docs


def add_embeddings_to_chroma():
    img_list = get_docs("/home/namvn/workspace/AI-model/multi-modal-search-app/data")
    # menu_dict = {}

    # menus = [
    #     {
    #         "text": "wild mushroom creame chicken - A variety of hand-picked mushrooms, cooked to perfection, mixed with velvety cream and served with freshly chopped scallions"
    #     },
    #     {
    #         "text": "creamy chocolate cheese cake- nested in a dark, moist brownie, sprinkled with choco chips"
    #     },
    #     {
    #         "text": "French onion soup - slow simmered sweet onions, topped with savory cheese and garnished with croutons"
    #     },
    #     {
    #         "text": "zesty salmon fillet - oven baked salmon fillet served with garlic-infused citrusy product kosher product"
    #     },
    #     {
    #         "text": "Nonna edetta's pizza - Fresh mozzarella, special homemade tomato sauce, veg mayonnaise and greens"
    #     },
    #     {
    #         "text": "tiramisu - layered Italian dessert made with refined marsala wine, rum and cocoa powder"
    #     },
    #     {
    #         "text": "Bruschetta - a classic Italian pasta dish with a creamy sauce and a crispy crust"
    #     },
    #     {
    #         "text": "burrata salad - a savory salad made with fresh basil, tomatoes, basil, oregano and more"
    #     },
    #     {
    #         "text": "carbonara pizza - a pizza filled with pasta, bread, Parmesan cheese, oregano and more"
    #     },
    #     {
    #         "text": "chicken parmesan - a savory pasta dish with chicken, Parmesan cheese, oregano and more"
    #     },
    #     {
    #         "text": "sheet pan panzanella - a savory pasta dish with a crispy crust and a creamy sauce"
    #     },
    # ]

    path_list = []
    for path in img_list:
        path_dict = {}
        path_dict["path"] = path
        path_list.append(path_dict)
    # print(path_list)
    coll.add(ids=[str(i) for i in range(len(img_list))],
         documents = img_list,
         metadatas = path_list,
         )


add_embeddings_to_chroma()
