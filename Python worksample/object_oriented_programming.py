'''
Polling places similation

Jinfei Zhu
'''

import sys
import random
import queue
import click
import util


class Voter(object):
    '''
    Represents a voter.
    Attributes: arrival_time, voting_voting duration, start_time
    '''
    def __init__(self, arrival_time, voting_duration):
        '''
        Constructor for the Voter class

        Input:
            arrival_time:(float) the time the voter arrives at the polls
            voting_duration: (float) the amount of time the voter takes to vote

        '''
        self.arrival_time = arrival_time
        self.voting_duration = voting_duration
        self.start_time = None
        self.departure_time = None

class Precinct(object):

    def __init__(self, name, hours_open, max_num_voters,
                 num_booths, arrival_rate, voting_duration_rate):
        '''
        Constructor for the Precinct class
        Input:
            name: (str) Name of the precinct
            hours_open: (int) Hours the precinct will remain open
            max_num_voters: (int) Number of voters in the precinct
            num_booths: (int) Number of voting booths in the precinct
            arrival_rate: (float) Rate at which voters arrive
            voting_duration_rate: (float) Lambda for voting duration
        '''

        self.name = name
        self.hours_open = hours_open
        self.max_num_voters = max_num_voters
        self.num_booths = num_booths
        self.arrival_rate = arrival_rate
        self.voting_duration_rate = voting_duration_rate
        self.time = 0


    def next_voter(self, percent_straight_ticket, straight_ticket_duration):
        '''
        Return next voter in the precinct.
        Input: 
            percent_straint_ticket: (float) percentage of straight ticket voters
            straight_ticket_duration: (int)the voting time for staight-ticket voters

        '''

        gap, voting_duration = util.gen_voter_parameters(self.arrival_rate, self.voting_duration_rate,\
                                                         percent_straight_ticket, straight_ticket_duration)
        self.time += gap
        return Voter(self.time, voting_duration)


    def simulate(self, percent_straight_ticket, straight_ticket_duration, seed):
        '''
        Simulate a day of voting
        Input:
            percent_straight_ticket: (float) Percentage of straight-ticket voters
                                     as a decimal between 0 and 1 (inclusive)
            straight_ticket_duration: (float) Voting duration for straight-ticket voters
            seed: (int) Random seed to use in the simulation
        Output:
            List of voters who voted in the precinct
        '''

        random.seed(seed)

        lst = []
        num_voted = 0

        Booths = VotingBooths(self.num_booths)

        while num_voted < self.max_num_voters:
            voter = self.next_voter(percent_straight_ticket, straight_ticket_duration)

            if self.time <= (self.hours_open * 60):
                if not Booths.full():
                    voter.start_time = voter.arrival_time
                else:
                    voter_done_departuretime = Booths.rm_voter()
                    voter.start_time = max(voter.arrival_time, voter_done_departuretime)
                voter.departure_time = voter.start_time + voter.voting_duration
                Booths.add_voter(voter)
                lst.append(voter)
                num_voted += 1
            else:
                break
        
        return lst


class VotingBooths(object):
    '''
    Encapsulate the behavior of the voting booths to keep track of 
    the voters who are currently occupying voting booths and 
    to determine which voter will depart next.
    '''
    
    def __init__(self, num_booths):
        '''
        Constructor for VotingBooths class.
            Input: num_booths: (int) the number of booth
        '''
        self.num_booths = num_booths
        self.__pq = queue.PriorityQueue(maxsize = self.num_booths)
        
    def full(self):
        '''
        Return true if the queue is full.
        '''
        return self.__pq.full()

    def add_voter(self, voter):
        '''
        Add a voter to the queue
        Input:
            voter: object in voter class
        '''
        self.__pq.put(voter.departure_time, block=False)
        
    def rm_voter(self):
        '''
        Remove and return a voter from the queue
        '''
        return self.__pq.get(block=False)


def find_avg_wait_time(precinct, percent_straight_ticket, ntrials, initial_seed=0):
    '''
    Simulates a precinct multiple times with a given percentage of straight-ticket
    voters. For each simulation, computes the average waiting time of the voters,
    and returns the median of those average waiting times.
    Input:
        precinct: (dictionary) A precinct dictionary
        percent_straight_ticket: (float) Percentage straight-ticket voters
        ntrials: (int) The number of trials to run
        initial_seed: (int) Initial seed for random number generator
    Output:
        The median of the average waiting times returned by simulating
        the precinct 'ntrials' times.
    '''


    waiting_time = []

    for i in range(ntrials):
        p = Precinct(precinct["name"], precinct["hours_open"], precinct["num_voters"], \
                precinct["num_booths"], precinct["arrival_rate"], precinct["voting_duration_rate"])
        voters = p.simulate(percent_straight_ticket, \
                                    precinct["straight_ticket_duration"], initial_seed)
        awg_wt = sum([v.start_time - v. arrival_time for v in voters]) / len(voters)
        waiting_time.append(awg_wt)
        initial_seed += 1

    sorted_lst = sorted(waiting_time)

    return sorted_lst[ntrials // 2]



def find_percent_split_ticket(precinct, target_wait_time, ntrials, seed=0):
    '''
    Finds the percentage of split-ticket voters needed to bound
    the (average) waiting time. 
    Input:
        precinct: (dictionary) A precinct dictionary
        target_wait_time: (float) The minimum waiting time
        ntrials: (int) The number of trials to run when computing 
                 the average waiting time                 
        seed: (int) A random seed
    Output:
        A tuple (percent_split_ticket, waiting_time) where:
        - percent_split_ticket: (float) The percentage of split-ticket
                                voters that ensures the average waiting time 
                                is above target_waiting_time
        - waiting_time: (float) The actual average waiting time with that
                        percentage of split-ticket voters
        If the target waiting time is infeasible, returns (1, None)
    '''

    percent_split_ticket = 1

    for percent in [x/10 for x in range(0, 11)]:
        percent_straight_ticket = 1 - percent
        avg_wt = find_avg_wait_time(precinct, percent_straight_ticket, ntrials, seed)
        if avg_wt > target_wait_time:
            percent_split_ticket = percent
            break
        
    if percent_split_ticket == 1.0 and avg_wt < target_wait_time:
        avg_wt = None
    
    return (percent_split_ticket, avg_wt)


