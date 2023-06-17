"""
2023 Semester 1 COMP90024 Cluster and Cloud Computing Assignment 1
Run the program using four computing cores.
mpiexec -n 4 python main.py twitter-data-small.json sal.json
"""

import argparse
import json
import os
import re
import time
from collections import Counter

from mpi4py import MPI
from tabulate import tabulate

states_dict = {
    "new south wales": "(nsw)",
    "victoria": "(vic.)",
    "queensland": "(qld)",
    "south australia": "(sa)",
    "western australia": "(wa)",
    "tasmania": "(tas.)",
    "northern territory": "(nt)",
    "australian capital territory": "(act)"
}

cities = ["sydney", "melbourne", "brisbane", "adelaide", "perth", "hobart", "darwin", "canberra"]
oter_cities = ["christmas island", "home island", "jervis bay", "norfolk island", "west island"]

pattern_data_author_id = re.compile("^\"author_id\": \"[0-9]+\",$")
pattern_author_id = re.compile("[0-9]+")
pattern_places_full_name = re.compile("^\"full_name\": \".+\",$")
pattern_author_location = re.compile("[A-Z].+[a-z]")


def main():
    start_time = time.time()

    args = get_args()
    locations_data = load_locations_data(args.sal_path)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    file_size = os.path.getsize(args.tweets_path)
    chunk_size = file_size / size

    chunk_start = int(chunk_size * rank)

    author_tweet_dict = {}

    count = 0  # The number of tweets processed by the current process

    with open(args.tweets_path, encoding="utf-8") as file:
        # Start reading the file from the start position.
        file.seek(chunk_start)

        # The following two variables control the manipulation of incomplete tweets.
        author_id = -1
        flag = True
        processed_chunk_size = 0

        # Keep reading the file until reaching the end position.
        while processed_chunk_size <= chunk_size:
            line = file.readline()

            if not line:
                break

            stripped_line = line.strip()

            if pattern_data_author_id.search(stripped_line):
                author_id = pattern_author_id.search(stripped_line).group(0)
                flag = False
            elif pattern_places_full_name.search(stripped_line):
                flag = True

                # The process knows the author id of the first tweet it should access.
                if author_id != -1:
                    count += 1
                    if pattern_author_location.search(stripped_line) is not None:
                        author_location = pattern_author_location.search(stripped_line).group(0)
                        process_single_tweet(author_tweet_dict, author_id, author_location, locations_data)

            processed_chunk_size += len(line.encode("utf-8"))

        # The current process only knows the author id of the last tweet it should access.
        if not flag:
            # Keep reading the file until finding the author's location.
            while True:
                stripped_line = file.readline().strip()

                if pattern_places_full_name.search(stripped_line):
                    count += 1
                    author_location = pattern_author_location.search(stripped_line).group(0)
                    process_single_tweet(author_tweet_dict, author_id, author_location, locations_data)
                    break

    print("The number of tweets processed by Rank", rank, ":", count)

    gathered_data = comm.gather(author_tweet_dict, root=0)

    if rank == 0:
        parallelization_end_time = time.time()
        if size > 1:
            print("\nApproximate Time for Parallel Computations:", parallelization_end_time - start_time, "seconds")

        author_tweet_dict = process_gathered_data(gathered_data)
        extract_info_and_print(author_tweet_dict)

        end_time = time.time()
        if size > 1:
            print("\nApproximate Time for Non-Parallel Computations:", end_time - parallelization_end_time, "seconds")
        print("\nTotal Time:", end_time - start_time, "seconds\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tweets_path", type=str)
    parser.add_argument("sal_path", type=str)
    return parser.parse_args()


def load_locations_data(sal_path):
    locations_json = open(sal_path, encoding="utf8")
    locations_data = json.load(locations_json)
    return locations_data


def process_single_tweet(author_tweet_dict, author_id, author_location, locations_data):
    """
    Process a single tweet.
    :param author_tweet_dict: a dictionary consisting of authors and their related information
    :param author_id: the author id of the current tweet
    :param author_location: the location of that author
    :param locations_data: the locations data provided by sal.json
    """
    city_codes_dict = {"1gsyd": 0, "2gmel": 0, "3gbri": 0, "4gade": 0, "5gper": 0, "6ghob": 0, "7gdar": 0, "8acte": 0,
                       "9oter": 0}

    # Check whether the author id is new.
    if author_id not in author_tweet_dict:
        author_tweet_dict[author_id] = {}
        author_tweet_dict[author_id]["city_codes"] = city_codes_dict

    # Check whether author location belongs to Great Other Territories.
    words = [word.strip() for word in author_location.split(',')]
    words = [word.lower() for word in words]

    oter_found = False
    for word in words:
        if word in oter_cities:
            author_tweet_dict[author_id]["city_codes"]["9oter"] += 1
            oter_found = True

    if not oter_found:
        for word in words:
            # Check whether the city name is provided.
            if word in cities:
                city_code = locations_data[word]["gcc"]
                if city_code in city_codes_dict:
                    author_tweet_dict[author_id]["city_codes"][city_code] += 1
                    break

            # Check whether the state is provided.
            for state in states_dict:
                if state in words:
                    suburb = word + " " + states_dict[state]
                    if suburb in locations_data:
                        city_code = locations_data[suburb]["gcc"]
                        if city_code in city_codes_dict:
                            author_tweet_dict[author_id]["city_codes"][city_code] += 1
                            break

            # Search for suburbs directly since the city and state are not provided.
            if word in locations_data:
                city_code = locations_data[word]["gcc"]
                if city_code in city_codes_dict:
                    author_tweet_dict[author_id]["city_codes"][city_code] += 1
                    break


def process_gathered_data(gathered_data):
    """
    Process the data gathered from all processes.
    :param gathered_data: a list containing dictionaries gathered from all processes
    :return: a single dictionary containing all authors and their relevant data
    """
    final_dict = {}

    for dict_collected_per_process in gathered_data:
        for author_id in dict_collected_per_process:
            # Handle the situation where the author already exists in the final dictionary.
            if author_id in final_dict:
                for city_code, num_tweets in dict_collected_per_process[author_id]["city_codes"].items():
                    final_dict[author_id]["city_codes"][city_code] += num_tweets
            else:
                final_dict[author_id] = dict_collected_per_process[author_id]

    return final_dict


def extract_info_and_print(author_tweet_dict):
    tweets_per_city_dict = {"1gsyd": 0, "2gmel": 0, "3gbri": 0, "4gade": 0, "5gper": 0, "6ghob": 0, "7gdar": 0,
                            "8acte": 0, "9oter": 0}

    for author in author_tweet_dict:
        # Calculate the number of tweets made by each author.
        author_tweet_dict[author]["statistics"] = {}
        author_tweet_dict[author]["statistics"]["total_tweets"] = sum(
            author_tweet_dict[author]["city_codes"].values())

        # Calculate the number of tweets made from each capital city.
        temp_dict = dict(Counter(author_tweet_dict[author]["city_codes"]))
        for city_code in temp_dict:
            tweets_per_city_dict[city_code] += temp_dict[city_code]

        # Calculate the number of capital cities the current author has tweeted in.
        author_tweet_dict[author]["statistics"]["unique_cities"] = sum(
            count != 0 for count in author_tweet_dict[author]["city_codes"].values())

    print_task_one(author_tweet_dict)
    print_task_two(tweets_per_city_dict)
    print_task_three(author_tweet_dict)


def print_task_one(author_tweet_dict):
    top_ten_tweeters_dict = dict(
        sorted(author_tweet_dict.items(), key=lambda item: item[1]["statistics"]["total_tweets"], reverse=True)[:10])

    results = []
    author_rank = 1
    for author_id in top_ten_tweeters_dict:
        author_rank_string = "#" + str(author_rank)
        results.append([author_rank_string, author_id, top_ten_tweeters_dict[author_id]["statistics"]["total_tweets"]])
        author_rank += 1

    print("\nTask 1:\n")
    print(tabulate(results, headers=["Rank", "Author Id", "Number of Tweets Made"], numalign="left"))


def print_task_two(tweets_per_city_dict):
    print("\nTask 2:\n")
    print(tabulate([
        ["1gsyd (Greater Sydney)", tweets_per_city_dict["1gsyd"]],
        ["2gmel (Greater Melbourne)", tweets_per_city_dict["2gmel"]],
        ["3gbri (Greater Brisbane)", tweets_per_city_dict["3gbri"]],
        ["4gade (Greater Adelaide)", tweets_per_city_dict["4gade"]],
        ["5gper (Greater Perth)", tweets_per_city_dict["5gper"]],
        ["6ghob (Greater Hobart)", tweets_per_city_dict["6ghob"]],
        ["7gdar (Greater Darwin)", tweets_per_city_dict["7gdar"]],
        ["8acte (Greater Canberra)", tweets_per_city_dict["8acte"]],
        ["9oter (Great Other Territories)", tweets_per_city_dict["9oter"]]
    ], headers=["Greater Capital City", "Number of Tweets Made"], numalign="left"))


def print_task_three(author_tweet_dict):
    top_ten_tweeters_dict = dict(sorted(author_tweet_dict.items(), key=lambda item:
    (item[1]["statistics"]["unique_cities"],
     item[1]["statistics"]["total_tweets"]), reverse=True)[:10])

    new_city_codes = ["gsyd", "gmel", "gbri", "gade", "gper", "ghob", "gdar", "acte", "oter"]
    for author_id in top_ten_tweeters_dict:
        top_ten_tweeters_dict[author_id]["city_codes"] = dict(zip(new_city_codes,
                                                                  list(top_ten_tweeters_dict[author_id][
                                                                           "city_codes"].values())))
    results = []
    author_rank = 1
    for author_id in top_ten_tweeters_dict:
        author_rank_string = "#" + str(author_rank)
        city_tweet_string = str(top_ten_tweeters_dict[author_id]["statistics"]["unique_cities"]) + "(#" \
                            + str(top_ten_tweeters_dict[author_id]["statistics"]["total_tweets"]) + "tweets - "

        for city_code in top_ten_tweeters_dict[author_id]["city_codes"]:
            if city_code == "oter":
                city_tweet_string += str(top_ten_tweeters_dict[author_id]["city_codes"][city_code]) + city_code + ")"
            else:
                city_tweet_string += str(top_ten_tweeters_dict[author_id]["city_codes"][city_code]) + city_code + ", "

        results.append([author_rank_string, author_id, city_tweet_string])
        author_rank += 1

    print("\nTask 3:\n")
    print(tabulate(results, headers=["Rank", "Author Id", "Number of Unique City Locations and #Tweets"],
                   numalign="left"))


if __name__ == "__main__":
    main()
