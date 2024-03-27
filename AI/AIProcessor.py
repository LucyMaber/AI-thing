data_class = {
    "GroundNews": {
        "tags": [],
        "bias": [],
    },
    "inyourarea": {"tags": [], "primary_place_concept_uri": []},
    "toot": {
        "tags": [],
        "language": [],
    },
    "web_page": {
        "tags": [],
        "lang": [],
    },
    "reddit": {
        "subreddit": [],
    },
    "wikidata": {
        "subclass_of": [],
        "part_of": [],
        "sex_or_gender": [],
        "sexual_orientation": [],
        "languages_spoken_written_or_signed": [],
        "political_alignment": [],
        "writing_language": [],
        "occupation": [],
        "ethnic_group": [],
        "political_ideology": [],
        "field_of_work": [],
        "member_of_political_party": [],
        "member_of": [],
        "country": [],
        "country_of_citizenship": [],
        "hashtags": [],
        "field_of_work": [],
        "hashtag": [],
        "instance_of": [],
        "movement": [],
        "occupation": [],
    },
}
def load_handler():
    # test if file exists
    if not os.path.isfile("data_class.json"):
        with open("data_class.json", "r") as f:
            data = json.load(f)
            for key in data.keys():
                if key in data_class:
                    data_class[key] = {}
                for subkey in data[key]:
                    data_class[key][subkey] = data[key][subkey]
                

# save on exit
import atexit
import datetime
import os
import random
import sys
import time
import traceback

import torch
from AI.LikesEstimator import (
    make_estimator,
    preprocess_estimator_data,
    train_model_likes_estimator,
)
from AI.tokenizer import make_classifier, train_model_classifier
import json

input_dim = 10000  # Size of the vocabulary
hidden_dim = 128
output_dim = 2  # Initial number of classes
num_heads = 4
num_layers = 2
dropout_prob = 0.1

# classifierShinigami_eye = make_classifier([True, False],"module/classifierShinigami_eye.pth")
# classifierReddit_subreddit = make_classifier(data_class["GroundNews"]["tags"],"module/classifierReddit_subreddit.pth")
# classifierGroundNews_tags = make_classifier(data_class["GroundNews"]["tags"],"module/classifierGroundNews_tags.pth")
# classifierGroundNews_bias = make_classifier(data_class["GroundNews"]["bias"],"module/classifierGroundNews_bias.pth")
# classifierToot_tags = make_classifier(data_class["toot"]["tags"],"module/classifierToot_tags.pth")
# classifierToot_language = make_classifier(data_class["toot"]["language"],"module/classifierToot_language.pth")
# classifierWeb_page_tags = make_classifier(data_class["web_page"]["tags"],"module/classifierWeb_page_tags.pth")
# classifierInyourarea_tags = make_classifier(data_class["inyourarea"]["tags"],"module/classifierInyourarea_tags.pth")
# classifierInyourarea_primary_place_concept_uri = make_classifier(data_class["inyourarea"]["primary_place_concept_uri"],"module/classifierInyourarea_primary_place_concept_uri.pth")
# classifierWikidataSubclass_of = make_classifier(data_class["wikidata"]["subclass_of"],"module/classifierWikidataSubclass_of.pth")
# classifierWikidataPart_of = make_classifier(data_class["wikidata"]["part_of"],"module/classifierWikidataPart_of.pth")
# classifierWikidataWriting_language = make_classifier(data_class["wikidata"]["writing_language"],"module/classifierWikidataWriting_language.pth")
# classifierWikidataOccupation = make_classifier(data_class["wikidata"]["occupation"],"module/classifierWikidataOccupation.pth")
# classifierWikidataEthnic_group = make_classifier(data_class["wikidata"]["ethnic_group"],"module/classifierWikidataEthnic_group.pth")
# classifierWikidataPolitical_ideology = make_classifier(data_class["wikidata"]["political_ideology"],"module/classifierWikidataPolitical_ideology.pth")
# classifierWikidataPoliticalAlignment = make_classifier(data_class["wikidata"]["political_alignment"],"module/classifierWikidataPoliticalAlignment.pth")
# classifierWikidataField_of_work = make_classifier(data_class["wikidata"]["field_of_work"],"module/classifierWikidataField_of_work.pth")
# classifierWikidataMember_of_political_party = make_classifier(data_class["wikidata"]["member_of_political_party"],"module/classifierWikidataMember_of_political_party.pth")
# classifierWikidataMember_of = make_classifier(data_class["wikidata"]["member_of"],"module/classifierWikidataMember_of.pth")
# classifierSexualOrientation = make_classifier(data_class["wikidata"]["sexual_orientation"],"module/classifierSexualOrientation.pth")
# classifierWikidataCountry = make_classifier(data_class["wikidata"]["country"],"module/classifierWikidataCountry.pth")
# classifierWikidataCountry_of_citizenship = make_classifier(data_class["wikidata"]["country_of_citizenship"],"module/classifierWikidataCountry_of_citizenship.pth")
# classifierWikidataHashtags = make_classifier(data_class["wikidata"]["hashtags"],"module/classifierWikidataHashtags.pth")
# classifierWikidataField_of_work = make_classifier(data_class["wikidata"]["field_of_work"],"module/classifierWikidataField_of_work.pth")
# classifierWikidataHashtag = make_classifier(data_class["wikidata"]["hashtag"],"module/classifierWikidataHashtag.pth")
# classifierWikidataInstance_of = make_classifier(data_class["wikidata"]["instance_of"],"module/classifierWikidataInstance_of.pth")
# classifierWikidataMovement = make_classifier(data_class["wikidata"]["movement"],"module/classifierWikidataMovement.pth")
# classifierWikidataOccupation = make_classifier(data_class["wikidata"]["occupation"],"module/classifierWikidataOccupation.pth")
# classifierWikidataSexOrGender = make_classifier(data_class["wikidata"]["sex_or_gender"],"module/classifierWikidataSexOrGender.pth")

# classifierWebPageLang= make_classifier(data_class["web_page"]["lang"],"module/classifierWebPageLang.pth")
# classifierTootLang= make_classifier(data_class["web_page"]["lang"],"module/classifierTootLang.pth")


# toot_followers_estomator = make_estimator("module/toot_followers_estomator.pth")
# toot_reblogs_estomator = make_estimator("module/toot_reblogs_estomator.pth")
# toot_replies_estomator = make_estimator("module/toot_replies_estomator.pth")



def load_handler():
    # Check if file exists
    if not os.path.isfile("data_class.json"):
        return False
    
    try:
        with open("data_class.json", "r") as f:
            data = json.load(f)
            # Clear existing data_class if needed
            # data_class.clear()
            for key in data:
                # Update data_class with the loaded data
                data_class[key].update(data[key])
        return True
    except (json.JSONDecodeError, IOError) as e:
        print("Error loading data:", e)
        return False

load_handler()


def exit_handler():
    with open("data_class.json", "w") as f:
        json.dump(data_class, f)
    # torch.save(classifierShinigami_eye.state_dict(),"module/classifierShinigami_eye.pth")
    # torch.save(classifierReddit_subreddit.state_dict(),"module/classifierReddit_subreddit.pth")
    # torch.save(classifierGroundNews_tags.state_dict(),"module/classifierGroundNews_tags.pth")
    # torch.save(classifierGroundNews_bias.state_dict(),"module/classifierGroundNews_bias.pth")
    # torch.save(classifierToot_tags.state_dict(),"module/classifierToot_tags.pth")
    # torch.save(classifierToot_language.state_dict(),"module/classifierToot_language.pth")
    # torch.save(classifierWeb_page_tags.state_dict(),"module/classifierWeb_page_tags.pth")
    # torch.save(classifierInyourarea_tags.state_dict(),"module/classifierInyourarea_tags.pth")
    # torch.save(classifierInyourarea_primary_place_concept_uri.state_dict(),"module/classifierInyourarea_primary_place_concept_uri.pth")
    # torch.save(classifierWikidataSubclass_of.state_dict(),"module/classifierWikidataSubclass_of.pth")
    # torch.save(classifierWikidataPart_of.state_dict(),"module/classifierWikidataPart_of.pth")
    # torch.save(classifierWikidataWriting_language.state_dict(),"module/classifierWikidataWriting_language.pth")
    # torch.save(classifierWikidataOccupation.state_dict(),"module/classifierWikidataOccupation.pth")
    # torch.save(classifierWikidataEthnic_group.state_dict(),"module/classifierWikidataEthnic_group.pth")
    # torch.save(classifierWikidataPolitical_ideology.state_dict(),"module/classifierWikidataPolitical_ideology.pth")
    # torch.save(classifierWikidataPoliticalAlignment.state_dict(),"module/classifierWikidataPoliticalAlignment.pth")
    # torch.save(classifierWikidataField_of_work.state_dict(),"module/classifierWikidataField_of_work.pth")
    # torch.save(classifierWikidataMember_of_political_party.state_dict(),"module/classifierWikidataMember_of_political_party.pth")
    # torch.save(classifierWikidataMember_of.state_dict(),"module/classifierWikidataMember_of.pth")
    # torch.save(classifierSexualOrientation.state_dict(),"module/classifierSexualOrientation.pth")
    # torch.save(classifierWikidataCountry.state_dict(),"module/classifierWikidataCountry.pth")
    # torch.save(classifierWikidataCountry_of_citizenship.state_dict(),"module/classifierWikidataCountry_of_citizenship.pth")
    # torch.save(classifierWikidataHashtags.state_dict(),"module/classifierWikidataHashtags.pth")
    # torch.save(classifierWikidataField_of_work,"module/classifierWikidataField_of_work.pth")
    # torch.save(classifierWikidataHashtag.state_dict(),"module/classifierWikidataHashtag.pth")
    # torch.save(classifierWikidataInstance_of.state_dict(),"module/classifierWikidataInstance_of.pth")
    # torch.save(classifierWikidataMovement.state_dict(),"module/classifierWikidataMovement.pth")
    # torch.save(classifierWikidataOccupation.state_dict(),"module/classifierWikidataOccupation.pth")
    # torch.save(classifierWikidataSexOrGender.state_dict(),"module/classifierWikidataSexOrGender.pth")
    # torch.save(toot_followers_estomator.state_dict(),"toot_followers_estomator.pth")
    # torch.save(toot_reblogs_estomator.state_dict(),"module/toot_reblogs_estomator.pth")
    # torch.save(toot_replies_estomator.state_dict(),"module/toot_replies_estomator.pth")


atexit.register(exit_handler)


async def faintly2(task, data):
    print("faintly2->",task, data.keys())
    if data is None:
        print("faintly2 missing data", task, None)
        return
    try:
        if random.randint(0, 5) == 3:
            exit_handler()
        global data_class
        is_pirate = None
        GroundNews_bias = None
        GroundNews_tags = None
        inyourarea_tags = None
        languages_spoken_written_or_signed = None
        country = None
        country_of_citizenship = None
        sexual_orientation = None
        sex_or_gender = None
        occupation = None
        ethnic_group = None
        writing_language = None
        political_alignment = None
        subclass_of = None
        political_ideology = None
        field_of_work = None
        member_of_political_party = None
        movement = None
        part_of = None
        subclass_of = None
        member_of = None
        country = None
        country_of_citizenship = None

        # if "toot" == task:
        #     print("toot", data.keys())
            # for tag in data["tags"]:
            # print("toot tag: ",tag.replace("https://tech.lgbt/tags/", ""))
        is_pirate = False
        if "is_pirate" in data.keys():
            is_pirate = True
        if "GroundNews" in data.keys():
            for bias in data["GroundNews"]["bias"]:
                if bias not in data_class["GroundNews"]["bias"]:
                    data_class["GroundNews"]["bias"].append(bias)
            GroundNews_bias = data["GroundNews"]["bias"]
            for tags in data["GroundNews"]["tags"]:
                if tags not in data_class["GroundNews"]["tags"]:
                    data_class["GroundNews"]["tags"].append(tags)
            GroundNews_tags = data["GroundNews"]["tags"]
        if "inyourarea" in data.keys():
            for tags in data["inyourarea"]["tags"]:
                if tags not in data_class["inyourarea"]["tags"]:
                    data_class["inyourarea"]["tags"].append(tags)
            inyourarea_tags = data["inyourarea"]["tags"]

        if "wikidata" in data.keys():
            if "wikidata_languages_spoken_written_or_signed" in data["wikidata"]:
                for item in data["wikidata"]["languages_spoken_written_or_signed"]:
                    if (
                        item
                        not in data_class["wikidata"][
                            "languages_spoken_written_or_signed"
                        ]
                    ):
                        data_class["wikidata"][
                            "languages_spoken_written_or_signed"
                        ].append(item)
                languages_spoken_written_or_signed = data["wikidata"][
                    "languages_spoken_written_or_signed"
                ]

            if "country" in data["wikidata"]:
                for item in data["wikidata"]["country"]:
                    if item not in data_class["wikidata"]["country"]:
                        data_class["wikidata"]["country"].append(item)
                country = data["wikidata"]["country"]

            if "country_of_citizenship" in data["wikidata"]:
                for item in data["wikidata"]["country_of_citizenship"]:
                    if item not in data_class["wikidata"]["country_of_citizenship"]:
                        data_class["wikidata"]["country_of_citizenship"].append(item)
                country_of_citizenship = data["wikidata"]["country_of_citizenship"]

            if "sexual_orientation" in data["wikidata"]:
                for item in data["wikidata"]["sexual_orientation"]:
                    if item not in data_class["wikidata"]["sexual_orientation"]:
                        data_class["wikidata"]["sexual_orientation"].append(item)
                sexual_orientation = data["wikidata"]["sexual_orientation"]
            if "sex_or_gender" in data["wikidata"]:
                for item in data["wikidata"]["sex_or_gender"]:
                    if item not in data_class["wikidata"]["sex_or_gender"]:
                        data_class["wikidata"]["sex_or_gender"].append(item)
                sex_or_gender = data["wikidata"]["sex_or_gender"]
            if "occupation" in data["wikidata"]:
                for item in data["wikidata"]["occupation"]:
                    if item not in data_class["wikidata"]["occupation"]:
                        data_class["wikidata"]["occupation"].append(item)
                occupation = data["wikidata"]["occupation"]
            if "ethnic_group" in data["wikidata"]:
                for item in data["wikidata"]["ethnic_group"]:
                    if item not in data_class["wikidata"]["ethnic_group"]:
                        data_class["wikidata"]["ethnic_group"].append(item)
                ethnic_group = data["wikidata"]["ethnic_group"]

            if "writing_language" in data["wikidata"]:
                for item in data["wikidata"]["writing_language"]:
                    if item not in data_class["wikidata"]["writing_language"]:
                        data_class["wikidata"]["writing_language"].append(item)
                writing_language = data["wikidata"]["writing_language"]

            if "political_alignment" in data["wikidata"]:
                for item in data["wikidata"]["political_alignment"]:
                    if item not in data_class["wikidata"]["political_alignment"]:
                        data_class["wikidata"]["political_alignment"].append(item)
                political_alignment = data["wikidata"]["political_alignment"]

            if "subclass_of" in data["wikidata"]:
                for item in data["wikidata"]["subclass_of"]:
                    if item not in data_class["wikidata"]["subclass_of"]:
                        data_class["wikidata"]["subclass_of"].append(item)
                subclass_of = data["wikidata"]["subclass_of"]

            if "political_ideology" in data["wikidata"]:
                for item in data["wikidata"]["political_ideology"]:
                    if item not in data_class["wikidata"]["political_ideology"]:
                        data_class["wikidata"]["political_ideology"].append(item)
                political_ideology = data["wikidata"]["political_ideology"]

            if "field_of_work" in data["wikidata"]:
                for item in data["wikidata"]["field_of_work"]:
                    if item not in data_class["wikidata"]["field_of_work"]:
                        data_class["wikidata"]["field_of_work"].append(item)
                field_of_work = data["wikidata"]["field_of_work"]

            if "member_of_political_party" in data["wikidata"]:
                for item in data["wikidata"]["member_of_political_party"]:
                    if item not in data_class["wikidata"]["member_of_political_party"]:
                        data_class["wikidata"]["member_of_political_party"].append(item)
                member_of_political_party = data["wikidata"][
                    "member_of_political_party"
                ]

            if "movement" in data["wikidata"]:
                for item in data["wikidata"]["movement"]:
                    if item not in data_class["wikidata"]["movement"]:
                        data_class["wikidata"]["movement"].append(item)
                movement = data["wikidata"]["movement"]

            if "part_of" in data["wikidata"]:
                for item in data["wikidata"]["part_of"]:
                    if item not in data_class["wikidata"]["part_of"]:
                        data_class["wikidata"]["part_of"].append(item)
                part_of = data["wikidata"]["part_of"]

            if "subclass_of" in data["wikidata"]:
                for item in data["wikidata"]["subclass_of"]:
                    if item not in data_class["wikidata"]["subclass_of"]:
                        data_class["wikidata"]["subclass_of"].append(item)
                subclass_of = data["wikidata"]["subclass_of"]

            if "member_of" in data["wikidata"]:
                for item in data["wikidata"]["member_of"]:
                    if item not in data_class["wikidata"]["member_of"]:
                        data_class["wikidata"]["member_of"].append(item)
                member_of = data["wikidata"]["member_of"]

        for temp_name in [
            "official_website",
            "subreddit",
            "Twitter_username",
            "Mastodon_address",
            "Instagram_username",
            "Medium_username",
            "Twitter_numeric_user_ID",
            "Parler_username",
            "Gab_username",
            "Truth_Social_username",
            "LinkedIn_personal_profile_ID",
            "Facebook_numeric_ID",
            "Tumblr_username",
            "YouTube_handle",
            "YouTube_channel_ID",
            "Flickr_user_ID",
            "Reddit_username",
            "Twitter_username",
            "web_feed_URL",
            "Facebook_ID",
            "Scinapse_author_ID",
            "Semantic_Scholar_author_ID",
            "Scopus_author_ID",
            "ORCID_iD",
            "Scinapse_author_ID",
            "CiteSeerX_person_ID",
            "SciProfiles_ID",
            "OpenAlex_ID",
            "OpenReview_net_profile_ID",
            "Scholars_Strategy_Network_ID",
            "ResearchGate_profile_ID",
            "LinkedIn_personal_profile_ID",
            "Crossref_journal_ID",
            "DOI",
        ]:
            if str(task) == temp_name:
                return
            # shinigami_eye = data["shinigami_eye"]["shinigami_eye"]
        description = None
        if task == "web_page":
            # print("twitter_name",data["web_page"]["twitter_name"] )
            # print("description",data["web_page"]["description"] )
            if data["web_page"]["body"] is not None:
                input_content: str = data["web_page"]["body"]
            elif data["web_page"]["content"] is not None:
                input_content: str = data["web_page"]["content"]
            elif data["web_page"]["textContent"] is not None:
                input_content: str = data["web_page"]["textContent"]
            elif data["web_page"]["articleBody_text"] is not None:
                input_content: str = data["web_page"]["articleBody_text"]
            else:
                return
        elif task == "tweet":
            # PREDICT LIKE
            # PREDICT REBLOG
            # PREDICT REPLY
            pass
        elif task == "toot":
            replies = data["toot"]["replies_count"]
            reblogs = data["toot"]["reblogs_count"]
            favourites = data["toot"]["favourites_count"]
            toot_account = data["toot_account"]
            following_count = toot_account["following_count"]
            followers_count = toot_account["followers_count"]
            created_at = datetime.datetime.strptime(
                data["toot"]["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            time_difference = (datetime.datetime.now() - created_at).total_seconds()
            input_content: str = data["toot"]["content"]
            pass
            # data_followers = preprocess_estimator_data(
            #     time_difference,
            #     followers_count,
            #     following_count,
            #     input_content,
            #     replies,
            # )
            # train_model_likes_estimator(toot_followers_estomator, data_followers)
            # data_reblogs = preprocess_estimator_data(
            #     time_difference,
            #     followers_count,
            #     following_count,
            #     input_content,
            #     reblogs,
            # )
            # train_model_likes_estimator(toot_reblogs_estomator, data_reblogs)
            # data_replies = preprocess_estimator_data(
            #     time_difference,
            #     followers_count,
            #     following_count,
            #     input_content,
            #     favourites,
            # )
            # train_model_likes_estimator(toot_replies_estomator, data_replies)
        elif task == "reddit_post":
            return
        else:
            return

        if type(input_content) == str:
            pass
        else:
            print("texts is not a string", type(input_content), task)
            return

        if "shinigami_eye" in data.keys():
            print("shinigami_eye", data["shinigami_eye"])
            pass
            # train_model_classifier(
            #     classifierShinigami_eye,
            #     input_content,
            #    [data["shinigami_eye"]["shinigami_eye"]],
            #     data["shinigami_eye"]["shinigami_eye"],
            # )
            pass

        if is_pirate is not None:
            # print("is_pirate", is_pirate)
            pass
        if GroundNews_bias is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierGroundNews_bias,
                #     input_content,
                #     GroundNews_bias,
                #     data_class["GroundNews"]["bias"],
                # )
            except:
                print("Error with GroundNews_bias training")
                pass
        if GroundNews_tags is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierGroundNews_tags,
                #     input_content,
                #     GroundNews_tags,
                #     data_class["GroundNews"]["tags"],
                # )
            except:
                print("Error with GroundNews_tags training")
                pass
        if inyourarea_tags is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierInyourarea_tags,
                #     input_content,
                #     inyourarea_tags,
                #     data_class["inyourarea"]["tags"],
                # )
            except:
                print("Error with inyourarea_tags training")
                pass
        if languages_spoken_written_or_signed is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataSubclass_of,
                #     input_content,
                #     languages_spoken_written_or_signed,
                #     data_class["wikidata"]["languages_spoken_written_or_signed"],
                # )
            except:
                print("Error with languages_spoken_written_or_signed training")
                pass
        if country is not None:
            try:
                print("Error with languages_spoken_written_or_signed training")
                pass
                # train_model_classifier(
                #     classifierWikidataCountry,
                #     input_content,
                #     country,
                #     data_class["wikidata"]["country"],
                # )
            except:
                print("Error with country traing")
                pass
        if sexual_orientation is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierSexualOrientation,
                #     input_content,
                #     sexual_orientation,
                #     data_class["wikidata"]["sexual_orientation"],
                # )
            except:
                print("Error with sexual_orientation training")
                pass
        if sex_or_gender is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataSexOrGender,
                #     input_content,
                #     sex_or_gender,
                #     data_class["wikidata"]["sex_or_gender"],
                # )
            except:
                print("Error with sex_or_gender training")
                pass
        if occupation is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataOccupation,
                #     input_content,
                #     occupation,
                #     data_class["wikidata"]["occupation"],
                # )
            except:
                print("Error with occupation training")
                pass
        if ethnic_group is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataEthnic_group,
                #     input_content,
                #     ethnic_group,
                #     data_class["wikidata"]["ethnic_group"],
                # )
            except:
                print("Error with ethnic_group training")
                pass
        if writing_language is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataWriting_language,
                #     input_content,
                #     writing_language,
                #     data_class["wikidata"]["writing_language"],
                # )
            except:
                print("Error with writing_language training")
                pass
        if political_alignment is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataPoliticalAlignment,
                #     input_content,
                #     political_alignment,
                #     data_class["wikidata"]["political_alignment"],
                # )
            except:
                print("Error with political_alignment training")
                pass
        if subclass_of is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataSubclass_of,
                #     input_content,
                #     subclass_of,
                #     data_class["wikidata"]["subclass_of"],
                # )
            except:
                print("Error with subclass_of training")
                pass
        if political_ideology is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataPolitical_ideology,
                #     input_content,
                #     political_ideology,
                #     data_class["wikidata"]["political_ideology"],
                # )
            except:
                print("Error with political_ideology training")
                pass
        if field_of_work is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataField_of_work,
                #     input_content,
                #     field_of_work,
                #     data_class["wikidata"]["field_of_work"],
                # )
            except:
                print("Error with field_of_work training")
                pass
        if member_of_political_party is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataMember_of_political_party,
                #     input_content,
                #     member_of_political_party,
                #     data_class["wikidata"]["member_of_political_party"],
                # )
            except:
                print("Error with member_of_political_party training")
                pass
        if movement is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataMovement,
                #     input_content,
                #     movement,
                #     data_class["wikidata"]["movement"],
                # )
            except:
                print("Error with movement training")
                pass
        if part_of is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataPart_of,
                #     input_content,
                #     part_of,
                #     data_class["wikidata"]["part_of"],
                # )
            except:
                print("Error with part_of training")
                pass
        if member_of is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataMember_of,
                #     input_content,
                #     member_of,
                #     data_class["wikidata"]["member_of"],
                # )
            except:
                print("Error with member_of training")
                pass
        if country_of_citizenship is not None:
            try:
                pass
                # train_model_classifier(
                #     classifierWikidataCountry,
                #     input_content,
                #     country_of_citizenship,
                #     data_class["wikidata"]["country_of_citizenship"],
                # )
            except:
                print("Error with country_of_citizenship training")
                pass
        # data["web_page"]["markdown_content"]
        # TEXT GAN
        # FFFS
    except Exception as e:
        # print line number
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = traceback.extract_tb(exc_tb)[-1][2]
        if data is None:
            print(exc_type, fname, exc_tb.tb_lineno, task, None)
            print("faintly2", e, task, None)
        else:
            print(exc_type, fname, exc_tb.tb_lineno, task, data.keys())
        print("faintly2", e)
