import multiprocessing
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.spatial import Delaunay
from skimage.draw import line
from statsmodels.distributions.empirical_distribution import ECDF
import itertools
from tqdm import tqdm
from operator import itemgetter
from collections import Counter


class Cell:
    """
    This Class defines the characteristics of a cell
    It gets binary image of the pixels, and id of the cell. Returns the object of cell
    """

    def __init__(self, labeled):
        """
        :param labeled: ndarray - Pixels of the cell is the label of the cell and the rest of the
        image is zero
        """
        labeled_image = labeled.copy()
        self.cell_id = np.unique(labeled_image)[1]
        labeled_image[labeled_image > 0] = 1
        wh = np.where(labeled_image > 0)
        pixs = np.array((wh[0] - np.mean(wh[0]), wh[1] - np.mean(wh[1])))
        ev, eigenvectors = np.linalg.eig(np.matmul(pixs, pixs.T))
        self.cell_center = np.array((np.int(np.mean(wh[1])), np.int(np.mean(wh[0]))))
        self.end1, self.end2 = end_finder((self.cell_center[1], self.cell_center[0]),
                                          labeled_image, eigenvectors[:, np.argmax(ev)])

        self.img_length = np.linalg.norm(np.array(self.end2) - np.array(self.end1))

        self.axis = self.cell_center - self.end1
        self.cell_length = np.linalg.norm(self.axis)

        self.neighbors = []


class Frame:
    """
    This class prepare and image as a frame with related objects in the frame which is bacteria cells
    """

    def __init__(self, image, labeled=None, max_dis_neighbor=85):
        """
        Here we initialize the basic characteristics of a frame
        :param image: ndarray - binary image of the cell
        :param labeled: ndarray - labeled pixels of each cell with its cell id number.
        :param max_dis_neighbor: maximum length that we consider a cell as a neighbor.
        """
        self.max_dis_neighbor = max_dis_neighbor
        self.cells = {}
        self.image = image
        if labeled is not None:
            cell_labels = np.sort(np.unique(labeled)[1:])
        else:
            labeled = label(image, connectivity=1)
            cell_labels = np.unique(labeled)[1:]

        self.cell_number = len(cell_labels)
        self.transfer_dict = {}
        self.inv_transfer_dict = {}
        self.labeled_image = np.zeros(np.shape(labeled)).astype(np.uint8)
        idx = 1
        for cn in cell_labels:
            self.transfer_dict[idx] = cn
            self.inv_transfer_dict[cn] = idx
            self.labeled_image[labeled == cn] = idx
            idx += 1

        for cn in range(1, idx):
            temp = self.labeled_image.copy()
            temp[temp != cn] = 0
            self.cells[cn] = Cell(temp)

        self.find_neighbors()

    def neigh_without_cell_cut(self, pind, pset, lis_lab, triangle, lab):
        """
        we find neighbors of cell number pind with Delaunay triangulation and if there is any overlap between the line
        from two neighbors and another cell we remove that neighbor as a candidate for neighbors.
        :param pind: index of cell integer
        :param pset: set of all centroid
        :param triangle: Delaunay triangulation between the cells
        :param lab: labeled image for cell numbers
        :return: neighbors with nan and without nan
        """
        neig = find_neighbors(pind, triangle)

        neigh_set = [[x, lis_lab[x]] for x in neig]
        for neighbor in neig:
            if np.linalg.norm((pset[pind] - pset[neighbor])) > self.max_dis_neighbor:
                neigh_set.remove([neighbor, lis_lab[neighbor]])

            else:
                line_pixel_set = list(lab[tuple(zip(line(int(pset[pind][1]), int(pset[pind][0]),
                                                         int(pset[neighbor][1]), int(pset[neighbor][0]))))][0])

                if {0, lis_lab[pind], lis_lab[neighbor]} != (set(np.unique(line_pixel_set)) | {0}):
                    neigh_set.remove([neighbor, lis_lab[neighbor]])

        return [x[1] for x in neigh_set]

    def find_neighbors(self):
        """
        This function find the neighbors of each cell
        """
        cent0 = np.array([cell.cell_center for cell in self.cells.values()])
        label_listed0 = np.array([num for num in self.cells.keys()])

        tri0 = Delaunay(cent0)

        for idx, cell in enumerate(self.cells.values()):
            cell.neighbors = self.neigh_without_cell_cut(idx, cent0, label_listed0,
                                                         tri0, self.labeled_image)


class NoSplitTrack:
    """
    This part we find the registration mapping for non split cells between two frames J and J_plus
    """
    def __init__(self, J, J_plus, lab_J=None, lab_J_plus=None, lambda_qua=1135.37370694, lambda_ovr=2979.6494527,
                 lambda_nst=2981.88035319, lambda_nfl=2903.09648722, target_window_len=50, growth_rate=1.3,
                 number_of_epoch=2000, temperature=50, dt=.995, epsilon=.005):
        """
        First we find the cells and related characteristic in each frame after that basic initialization
        :param J: ndarray - image at frame (t)
        :param J_plus: ndarray - image at frame (t + \Delta t)
        :param lab_J: ndarray - labeled each pixel of cells in frame J [optional]
        :param lab_J_plus: ndarray - labeled each pixel of cells in frame J_plus [optional]
        :param lambda_qua: float - coefficient of qual
        :param lambda_ovr: float - coefficient of overlap penalty
        :param lambda_nst: float - coefficient of neighbors stability penalty
        :param lambda_nfl: float - coefficient of neighbors flip penalty
        :param target_window_len: int - half of the length of the target window
        :param growth_rate: float - approximated growth rate between two frames
        :param number_of_epoch: int - number of epoch
        :param temperature: float - temperature of the system
        :param dt: float - decreasing rate of the temperature of the system
        :param epsilon: float - small value to stop the algorithm
        """
        self.J = Frame(J, lab_J)
        self.J_plus = Frame(J_plus, lab_J_plus)
        self.tw_len = target_window_len
        self.growth_rate = growth_rate
        self.small_number = 10**(-9)
        self.number_of_cells = len(self.J.cells)
        self.flip_threshold = -.8
        self.lambda_qua = lambda_qua
        self.lambda_ovr = lambda_ovr
        self.lambda_nst = lambda_nst
        self.lambda_nfl = lambda_nfl
        self.number_of_epoch = number_of_epoch
        self.temperature = temperature
        self.dt = dt
        self.epsilon = epsilon
        # minimum number of the epoch that algorithm supposed to investigate
        self.min_num_epoch = 100
        # This value find the average value of this number of epoch
        self.look_back = 30

        # Tracker of the penalties and cost function
        self.epoch_cost = []
        self.qual_cost = []
        self.overlap_cost = []
        self.nstab_cost = []
        self.nflip_cost = []

        # registration would be our mapping after optimization
        self.registration = {}
        # This is all the three measures for qual pseudo likelihood
        self.measure_dictionary = {}
        # CDFs for pseudo likelihood
        self.KIN = None
        self.DIS = None
        self.ROT = None
        # This N x N_+ table represent all the pseudo likelihood for (b, f(b)) and its loglikelihood
        self.likelihood_table = np.ones((len(self.J.cells) + 1,
                                         len(self.J_plus.cells) + 1)) * self.small_number
        self.loglikelihood_table = None
        # this is the number of the neighbors in the frame, equivalent to average number of the neighbors
        self.num_neighbors = 0
        # we save the possible cases for registration in this dictionary
        self.target_window = {}
        img_size = np.shape(self.J.image)
        for cell in self.J.cells.values():
            self.num_neighbors += len(cell.neighbors)
            self.target_window[cell.cell_id] = []
            top = cell.cell_center[1]-self.tw_len
            if top < 0:
                top = 0
            bot = cell.cell_center[1]+self.tw_len
            if bot >= img_size[0]:
                bot = img_size[0] - 1
            left = cell.cell_center[0]-self.tw_len
            if left < 0:
                left = 0
            right = cell.cell_center[0]+self.tw_len
            if right >= img_size[1]:
                right = img_size[1] - 1
            poss_cells = np.unique(self.J_plus.labeled_image[top:bot, left:right])[1:]
            for pcn in poss_cells:
                if cell.cell_center[0]-self.tw_len <= \
                        self.J_plus.cells[pcn].cell_center[0] <= \
                        cell.cell_center[0]+self.tw_len and \
                        cell.cell_center[1]-self.tw_len <= \
                        self.J_plus.cells[pcn].cell_center[1] <= \
                        cell.cell_center[1]+self.tw_len:
                    self.target_window[cell.cell_id].append(pcn)

    def initialization(self):
        """
        This function initialize the registration mapping 'f' as described in the paper
        """
        # First we make the likelihood table for all possible cases.
        self.make_likelihood_table()
        # In this for loop we initialize the registration with minimum possible likelihood of the
        # combination of neighbors of each registration. For the details you can look at the paper
        # for initialization of the registration mapping.
        for idx in self.J.cells.keys():
            temp = np.zeros((2, len(self.target_window[idx])))
            i = 0
            for k in self.target_window[idx]:
                temp[0, i] = k
                temp[1, i] = self.prob_with_neighbors(idx, k)

                i += 1

            self.registration[idx] = [int(temp[0, np.argmax(temp[1, :])])]

    def measure_finder(self):
        """
        This function gets values and returns dictionary of the measures defined for qual
        between cells of J and J_+
        """
        for idx, cell in self.J.cells.items():
            # for each cell make a dictionary of all
            temp_dic = {}
            for k in self.target_window[idx]:
                temp_dic[k] = [kinetic(cell.cell_center, self.J_plus.cells[k].cell_center),
                               distortion(cell.axis, self.J_plus.cells[k].axis, self.growth_rate),
                               rotation(cell.axis, self.J_plus.cells[k].axis)]

            self.measure_dictionary[idx] = temp_dic.copy()

    def make_cdf(self, nall_=True, nall_val=2):
        """
        This function returns set of length that we will use to find the probability of each assignment
        :param nall_: boolean - not all - True if we wanted to consider a portion of the sets as output
        :param nall_val: int - length of the portion of the output
        """
        self.measure_finder()
        m1_set = []
        m2_set = []
        m3_set = []
        for key1, val1 in self.measure_dictionary.items():
            if nall_:
                temp_1 = []
                temp_2 = []
                temp_3 = []
                for key2, val2 in val1.items():
                    temp_1.append(val2[0])
                    temp_2.append(val2[1])
                    temp_3.append(val2[2])
                for s in np.sort(temp_1)[:nall_val]:
                    m1_set.append(s)
                for s in np.sort(temp_2)[:nall_val]:
                    m2_set.append(s)
                for s in np.sort(temp_3)[:nall_val]:
                    m3_set.append(s)

            else:
                for key2, val2 in val1.items():
                    m1_set.append(val2[0])
                    m2_set.append(val2[1])
                    m3_set.append(val2[2])

        self.KIN = ECDF(m1_set)
        self.DIS = ECDF(m2_set)
        self.ROT = ECDF(m3_set)

    def pseudo_likelihood(self, m_vec):
        """
        this function finds the pseudo likelihood related to each triplet of (vkin, vdix, vrot)
        :param m_vec: (float, float, float) - triplet of (vkin, vdix, vrot)
        :return: float - PSL(b, f_b)
        """
        PSL_b_fb = (1 - self.KIN(m_vec[0])) * (1 - self.DIS(m_vec[1])) * (1 - self.ROT(m_vec[2]))
        return self.small_number if PSL_b_fb == 0 else PSL_b_fb

    def make_likelihood_table(self):
        """
        This function make the table of the likelihood from the cells in B to the cells in B_+
        """
        self.make_cdf()
        for key_j, val_j in self.measure_dictionary.items():
            for key_j_plus, val_j_plus in val_j.items():
                self.likelihood_table[key_j, key_j_plus] = self.pseudo_likelihood(val_j_plus)

        self.loglikelihood_table = np.log(self.likelihood_table)

    def single_pwn(self, b, f_b, one_comb):
        """
        here we compute likelihood for initialization of mapping f(b) = f_b just for one combination of outputs
        of f(b) = f_b
        :param b: cell number in I_0, integer
        :param f_b: cell number in I_1 which we find P(f(b)), integer
        :param one_comb: one combination for neighbors of the cell b
        :return: Probability for one combination of f(b) = f_b
        """
        prob = self.likelihood_table[b, f_b]

        for y in one_comb:
            # Here we test if the neighbors will stay in the second image as neighbors
            num_same_nei = 0
            ln0 = list(np.intersect1d(self.J.cells[b].neighbors, self.J.cells[y[0]].neighbors))
            ln1 = list(np.intersect1d(self.J_plus.cells[f_b].neighbors, self.J_plus.cells[y[1]].neighbors))
            test_pd = list(itertools.product(ln0, ln1))

            base_vec0 = self.J.cells[b].cell_center - self.J.cells[y[0]].cell_center
            base_vec1 = self.J_plus.cells[f_b].cell_center - self.J_plus.cells[y[1]].cell_center

            vlogic = True
            # here we test if they flip
            for k in test_pd:
                if k in one_comb:
                    num_same_nei += 1

                    temp_vec0 = self.J.cells[b].cell_center - self.J.cells[k[0]].cell_center
                    temp_vec1 = self.J_plus.cells[f_b].cell_center - self.J_plus.cells[k[1]].cell_center

                    if np.cross(base_vec0, temp_vec0) * np.cross(base_vec1, temp_vec1) < 0:
                        vlogic = False

            if y[1] in self.target_window[y[0]] and num_same_nei == len(ln0) and vlogic:
                prob *= self.likelihood_table[y[0], y[1]]

            else:
                prob *= self.small_number

        return prob**(1/(len(one_comb) + 1))

    def prob_with_neighbors(self, b, f_b):
        """
        here we compute likelihood for cell number f(b) = f_b all the combination
        of neighbors
        :param b: int - cell number in J, integer
        :param f_b: int - cell number in J_plus which we find P(f(b)), integer
        :return: list of possible probabilities for the cell b with different neighbors
        """
        # Here we find all possible match of neighbors of the cell 'b' to the neighbors
        # of the cell 'f(b)'. [(b_1, f(b_1), (b_2, f(b_2)), ..., (b_n, f(b_n)]
        possible_list = [list(zip(x, self.J_plus.cells[f_b].neighbors)) for x in
                         itertools.permutations(self.J.cells[b].neighbors,
                                                int(np.min((len(self.J_plus.cells[f_b].neighbors),
                                                            len(self.J.cells[b].neighbors)))))]
        # We find the semi likelihood related all possible cases possible_list.
        prob_list_acell = list(map(lambda x: [self.single_pwn(b, f_b, x), x, [b, f_b]], possible_list))

        # We find the the maximum semi likelihood assigned to the possible cases.
        max_prob = np.where(np.array(prob_list_acell, dtype=object)[:, 0] ==
                            np.max(np.array(prob_list_acell, dtype=object)[:, 0]))[0]

        # Choose on of the maximum cases
        prob_rc = np.random.choice(max_prob)

        # return the semi likelihood related to this (b, f_b)
        return prob_list_acell[prob_rc][0]

    def qual_penalty(self):
        """
        This function evaluate the qual penalty for the self.registration at the moment
        """
        return - np.sum(list(map(lambda x: self.loglikelihood_table[x, self.registration[x]],
                                 self.registration.keys()))) / self.number_of_cells

    def overlap_penalty(self):
        """
        overlap penalty for self.registration at the moment
        :return: int - number of overlap of prediction
        """
        _, occurrence = np.unique(list(self.registration.values()), return_counts=True)
        return (sum(occurrence) - len(occurrence)) / self.number_of_cells

    def hmn_not_neighbor(self, b):
        """
        looks at cells and count how many neighbors of the cells will not stay neighbors in the registration
        :param b: cell number in J
        :return: ratio of the cell the won't stay as neighbor in the second image
        """
        return np.sum(list(map(lambda x: self.registration[x] not in self.J_plus.
                               cells[self.registration[b][0]].neighbors,
                               self.J.cells[b].neighbors))) / len(self.J.cells[b].neighbors)

    def neighbors_stability_penalty(self):
        """
        :return: float - neighbor stability penalty value for the registration
        """
        return np.sum(list(map(lambda x: self.hmn_not_neighbor(x), self.registration.keys()))) / \
               (2 * self.number_of_cells)

    def three_vec_flip(self, b, b_neis):
        """
        This function look at the order of the vector from center b to two other center in b_neis
        if we do not have flip it will return 0, otherwise, they flipped and it will return 1
        :param b: integer - cell number
        :param b_neis: pair of two integers - two neighbors to find our we have flip or not
        :return: 0 if vectors don't flip, 1 if they flipped
        """
        base_vec0 = self.J.cells[b].cell_center - self.J.cells[b_neis[0]].cell_center
        base_vec1 = self.J_plus.cells[self.registration[b][0]].cell_center - \
                    self.J_plus.cells[self.registration[b_neis[0]][0]].cell_center

        temp_vec0 = self.J.cells[b].cell_center - self.J.cells[b_neis[1]].cell_center
        temp_vec1 = self.J_plus.cells[self.registration[b][0]].cell_center - \
                    self.J_plus.cells[self.registration[b_neis[1]][0]].cell_center

        if np.linalg.norm(temp_vec1) * np.linalg.norm(base_vec1) == 0:
            return 0

        if np.linalg.norm(temp_vec1) * np.linalg.norm(base_vec1) == 0:
            return 0

        deg0 = np.dot(base_vec0, temp_vec0) / (np.linalg.norm(temp_vec0) * np.linalg.norm(base_vec0))
        deg1 = np.dot(base_vec1, temp_vec1) / (np.linalg.norm(temp_vec1) * np.linalg.norm(base_vec1))

        if deg0 < self.flip_threshold:
            return 0

        if deg1 < self.flip_threshold:
            return 0

        if np.cross(base_vec0, temp_vec0) * np.cross(base_vec1, temp_vec1) < 0:
            return 1
        else:
            return 0

    def neighbors_flip_penalty(self):
        """
        this function counts all of the flips of the vectors
        :return: float - neighbors flip penalty for the registration at the moment
        """
        flip_num = 0
        for k in self.registration.keys():
            if len(self.J.cells[k].neighbors) > 1:
                comb = itertools.combinations(self.J.cells[k].neighbors, 2)
                flip_num += np.sum(list(map(lambda x: self.three_vec_flip(k, x), comb)))

        return flip_num / (4 * self.num_neighbors)

    def evaluate_cost_function(self):
        """
        :return: float - cost function value for the registration at the moment
        """
        return self.lambda_qua * self.qual_penalty() + self.lambda_ovr * self.overlap_penalty() + \
               self.lambda_nst * self.neighbors_stability_penalty() + \
               self.lambda_nfl * self.neighbors_flip_penalty()

    def boltzmann_machine_optimization(self):
        """
        This finds the best registration to minimize the cost function
        """
        # First assign the initial value for the registration
        self.initialization()
        self.epoch_cost.append(self.evaluate_cost_function())
        old_cost = self.epoch_cost[0]
        for epoch_counter in tqdm(range(self.number_of_epoch)):
            self.qual_cost.append(self.qual_penalty())
            self.overlap_cost.append(self.overlap_penalty())
            self.nstab_cost.append(self.neighbors_stability_penalty())
            self.nflip_cost.append(self.neighbors_flip_penalty())

            for idx in self.registration.keys():

                temp = self.registration[idx][0]

                p_case = self.target_window[idx].copy()
                p_case.remove(self.registration[idx][0])

                if len(p_case) == 0:
                    continue

                self.registration[idx] = [np.random.choice(p_case)]

                new_cost = self.evaluate_cost_function()

                delta_cost = new_cost - old_cost
                if delta_cost < 0:
                    old_cost = new_cost

                else:
                    u = np.random.uniform()
                    if u < np.exp(-delta_cost / self.temperature):
                        old_cost = new_cost
                    else:
                        self.registration[idx] = [temp]

            self.epoch_cost.append(self.evaluate_cost_function())
            self.temperature *= self.dt
            if len(self.epoch_cost) > self.min_num_epoch and \
                    np.sum(np.abs(np.diff(self.epoch_cost[-self.look_back:]))) < \
                    self.epsilon * self.epoch_cost[-self.min_num_epoch]:
                break


class SplitTrack:

    def __init__(self, J, J_plus, lab_J=None, lab_J_plus=None, lambda_cen=25, lambda_siz=.05,
                 lambda_ang=500, lambda_gap=.01, lambda_rat=.0001, lambda_over=500, lambda_rnk=.05,
                 epoch_num=5000, temperature=1000, dt=.995, epsilon=.005, growth_rate=1.05):
        """

        :param J: ndarray - image at frame (t)
        :param J_plus: ndarray - image at frame (t + \Delta t)
        :param lab_J: ndarray - labeled each pixel of cells in frame J [optional]
        :param lab_J_plus: ndarray - labeled each pixel of cells in frame J_plus [optional]
        :param lam1: float - first coefficient of the children penalty
        :param lam2: float - second coefficient of the children penalty
        :param lam3: float - third coefficient of the children penalty
        :param lam4: float - fourth coefficient of the children penalty
        :param epoch_num: int - maximum number of epochs
        :param temperature: float - temperature of the system
        :param dt: float - decreasing rate of the temperature
        :param epsilon: float - small number to stop the algorithm when it convergece to
        an value
        """
        # first and second frame that we want to find the split which they are from class of Frame
        self.J = Frame(J, lab_J)
        self.J_plus = Frame(J_plus, lab_J_plus)
        self.growth_rate = growth_rate

        # lambda one to four coefficients of the cost function
        self.lambda_gap = lambda_gap
        self.lambda_rat = lambda_rat
        self.lambda_over = lambda_over
        self.lambda_rnk = lambda_rnk
        self.lambda_cen = lambda_cen
        self.lambda_ang = lambda_ang
        self.lambda_siz = lambda_siz
        # number of the epoch
        self.number_of_epoch = epoch_num
        # temperature
        self.t = temperature
        # ratio to change the temprature
        self.dt = dt
        # epsilon value to stop the process
        self.epsilon = epsilon
        # number of cells that has been split
        self.split_num = self.J_plus.cell_number - self.J.cell_number
        # number of small children to consider for possible cases
        # self.small_number = 4 * self.split_num + 10
        self.small_number = max(int(self.J_plus.cell_number * 2 / 3), 4 * self.split_num + 15)
        # number of pairs
        # if self.split_num * 4 + 4 < 40:
        #     self.number_of_pairs = int(4 * self.split_num + 4)
        # elif self.split_num * 3 + 4 < 40:
        #     self.number_of_pairs = int(3 * self.split_num + 4)
        # else:
        #     self.number_of_pairs = int(2 * self.split_num + 4)
        self.number_of_pairs = self.split_num

        # cost of the pairs for each step
        self.cost = []
        # all cells that may be one of the children
        self.possible_cases = []
        # first coordinate and second coordinate of self.pairs
        self.pairs_col1 = []
        self.pairs_col2 = []
        # the target pairs that we want to find
        self.pairs = []
        # all parameters that we want to store them
        self.overlaps = []
        self.ratio_cost = []
        self.distance_cost = []
        self.length_ratio_cost = []
        self.epoch_cost = []
        self.new_cost = []
        self.len_cost = []
        self.par_cost = []
        # dictionary of the choice to match for each cell
        self.choices = {}
        # dictionary of parents of children
        self.possible_parents = {}
        self.pairs_with_parent = {}
        # this is the set of all possible pairs
        self.all_pairs = []
        self.pairs_temp = []
        # this is the radius that we won't accept any cell with the center of the base cell farther than this
        self.neighbor_radius = 45
        # this variable count the number of times initial value reapeat and if it is
        # more than 5 we stop the process.
        self.while_counter = 0
        # these two are the neighbors for frame 0 and 1
        self.neigh0 = {}
        self.neigh1 = {}
        # length of the minimum cell
        self.minimum_length = min([np.linalg.norm(np.array(self.J_plus.cells[y].end1) -
                                                  np.array(self.J_plus.cells[y].end2))
                                   for y in self.J_plus.cells.keys()])
        # threshold for the parameters
        self.deviate_threshold = 10
        self.distance_threshold = 30
        self.ratio_threshold = .2
        self.length_threshold = 20
        self.look_parents = 25

    def possible_maker(self):
        """
        This function finds label of all small_number smallest cells and consider them as
        possible case for children
        """
        s_cases = small_cell_finder(self.J_plus, self.small_number)
        for case in s_cases:
            self.possible_cases.append(case[0])

    def choice_maker(self):
        """
        This function make possible pairs for a children as sibling. We put the possible sibling in
        the choices dictionary
        """
        for c in self.possible_cases:
            self.choices[c] = []
            p_cases = self.possible_cases.copy()
            p_cases.remove(c)
            for pc in p_cases:
                if np.linalg.norm(np.array(self.J_plus.cells[c].cell_center) -
                                  np.array(self.J_plus.cells[pc].cell_center)) < self.neighbor_radius:
                    self.choices[c].append(pc)

    def all_pairs_maker(self):
        """
        This function pairs possible sibling. If some penalties of the possible siblings are greater
        than the threshold we do not consider them as a pair of potential children. After that for each
        pair of potential children we try to find a potential parents for them. We put the potential parent
        to possible_parent dictionary. If we could not find a parent for a pair of children we do not add
        that pair to possible children list (a.k.a. all_pairs).
        """
        self.choice_maker()
        for key, val in self.choices.items():
            for cell in val:
                if key < cell:
                    penalties = self.pair_values(key, cell)
                    len1 = (self.minimum_length - np.linalg.norm(np.array(self.J_plus.cells[key].end1) -
                                                                 np.array(self.J_plus.cells[key].end2)))
                    len2 = (self.minimum_length - np.linalg.norm(np.array(self.J_plus.cells[cell].end1) -
                                                                 np.array(self.J_plus.cells[cell].end2)))
                    if len1 + len2 < self.length_threshold and penalties[0] < self.deviate_threshold and \
                            penalties[1] < self.deviate_threshold and penalties[2] < self.distance_threshold and \
                            penalties[3] < self.ratio_threshold:

                        self.possible_parents[(key, cell)] = self.which_parent(key, cell)
                        if self.possible_parents[(key, cell)]:
                            self.all_pairs.append((key, cell))

    def which_parent(self, c1, c2):
        """
        This function return a potential parent between all possible parents for two children
        c1 and c2. We accept the parent with lowest parent penalty as potential parent between
        all possiblities. If we could not find one we return an empty set.
        :param c1: int - label of the one children
        :param c2: int - label of the other one children
        :return: int or [] - label of potential parent. If no parent [].
        """
        mid_point = (np.array(self.J_plus.cells[c1].cell_center) +
                     np.array(self.J_plus.cells[c2].cell_center)) / 2

        cells = []

        for c in self.J.cells:
            if np.linalg.norm(mid_point - np.array(self.J.cells[c].cell_center)) < self.look_parents:
                cells.append((c, self.parent_cost(c1, c2, c, (self.lambda_cen, self.lambda_ang, self.lambda_siz))))

        return sorted(cells, key=lambda x: x[1])[0][0] if cells else []

    def parent_cost(self, c1, c2, p, lam=None):
        """
        This function finds the cost values for relation between parents and children
        :param c1: integer - cell number for one of children in J_+
        :param c2: integer - cell number for another children in J_+
        :param p: integer - cell number for possible parent of children in I_0
        :param lam: triple of float - coefficient values for cost parameters
        :return: if lam=None returns all parameters separately, else multiply lam with parameters and add them
        """
        mid_point = (np.array(self.J_plus.cells[c1].cell_center) +
                     np.array(self.J_plus.cells[c2].cell_center)) / 2

        center_distance = np.linalg.norm(mid_point - np.array(self.J.cells[p].cell_center))

        middle_vec = np.array(self.J_plus.cells[c1].cell_center) - np.array(self.J_plus.cells[c2].cell_center)
        c1_vec = np.array(self.J_plus.cells[c1].cell_center) - np.array(self.J_plus.cells[c2].end1)
        c2_vec = np.array(self.J_plus.cells[c2].cell_center) - np.array(self.J_plus.cells[c2].end2)
        par_vec = np.array(self.J.cells[p].end2) - np.array(self.J.cells[p].end1)

        if np.linalg.norm(par_vec) == 0 or np.linalg.norm(c1_vec) == 0 or np.linalg.norm(c2_vec) == 0:
            angle_cost = 0
        else:
            angle_cost = np.abs(np.cross(par_vec, c1_vec)/(np.linalg.norm(par_vec) * np.linalg.norm(c1_vec))) + \
                         np.abs(np.cross(par_vec, c2_vec)/(np.linalg.norm(par_vec) * np.linalg.norm(c2_vec))) + \
                         np.abs(np.cross(par_vec, middle_vec)/(np.linalg.norm(middle_vec) * np.linalg.norm(par_vec)))

        siz_cost = np.abs(self.J_plus.cells[c1].cell_length + self.J_plus.cells[c2].cell_length -
                          (self.growth_rate * self.J.cells[p].cell_length))

        if not lam:
            return center_distance, angle_cost, siz_cost
        else:
            return lam[0] * center_distance + lam[1] * angle_cost + lam[2] * siz_cost

    def initialize_without_neighbor(self):
        """
        This function initialize values for the boltzmann machine. First make possible
        choices. After that make a random pairs
        :return:
        """
        self.possible_maker()
        self.all_pairs_maker()
        for _ in range(self.number_of_pairs):
            p_case = self.all_pairs.copy()
            for pair in self.pairs:
                p_case.remove(pair)
            if len(p_case) == 0:
                # print('this file has emptied p_case', self.J_plus.file_name)
                break
            temp_idx = np.random.choice(len(p_case))
            self.pairs.append(p_case[temp_idx])
            self.pairs_col1.append(p_case[temp_idx][0])
            self.pairs_col2.append(p_case[temp_idx][1])

    def pair_values(self, c1, c2):
        """
        This function compute some cost values for two cells c1 and c2
        :param c1: int - cell number one
        :param c2: int - cell number two
        :return: deviate penalty c1, deviate penalty c2, distance penalty two cells, ratio penalty two cells
        """
        possible_cases = np.array(list(itertools.product((self.J_plus.cells[c1].end1,
                                                          self.J_plus.cells[c1].end2),
                                                         (self.J_plus.cells[c2].end1,
                                                          self.J_plus.cells[c2].end2))))
        dis_list = []
        for i in range(len(possible_cases)):
            dis_list.append(np.linalg.norm(possible_cases[i][0] - possible_cases[i][1]))

        low_dis = possible_cases[np.argmin(dis_list)]

        dis0 = np.cross(np.array(self.J_plus.cells[c1].cell_center) - np.array(self.J_plus.cells[c2].cell_center),
                        np.array(self.J_plus.cells[c2].cell_center) - np.array(low_dis[0])) / \
               np.linalg.norm(np.array(self.J_plus.cells[c1].cell_center) -
                              np.array(self.J_plus.cells[c2].cell_center))

        dis1 = np.cross(np.array(self.J_plus.cells[c1].cell_center) - np.array(self.J_plus.cells[c2].cell_center),
                        np.array(self.J_plus.cells[c2].cell_center) - np.array(low_dis[1])) / \
               np.linalg.norm(np.array(self.J_plus.cells[c1].cell_center) -
                              np.array(self.J_plus.cells[c2].cell_center))

        length_ratio = abs(2 - (np.linalg.norm(np.array(self.J_plus.cells[c1].end1) -
                                               np.array(self.J_plus.cells[c1].end2)) /
                                np.linalg.norm(np.array(self.J_plus.cells[c2].end1) -
                                               np.array(self.J_plus.cells[c2].end2))) -
                           (np.linalg.norm(np.array(self.J_plus.cells[c2].end1) -
                                           np.array(self.J_plus.cells[c2].end2)) /
                            np.linalg.norm(np.array(self.J_plus.cells[c1].end1) -
                                           np.array(self.J_plus.cells[c1].end2))))

        return abs(dis0), abs(dis1), np.linalg.norm(np.array(low_dis[0]) - np.array(low_dis[1])), length_ratio

    def rank_cost(self):
        return sum(np.abs([(self.minimum_length - np.linalg.norm(np.array(self.J_plus.cells[y].end1) -
                                                                 np.array(self.J_plus.cells[y].end2)))
                           for y in self.pairs_col2 + self.pairs_col1])) / (2 * len(self.pairs))

    def total_parent_cost(self):
        """
        This function evaluate the total cost penalty for the parent cost
        :return: float - parent cost
        """
        return np.sum(list(map(lambda x, y: self.parent_cost(x, y, self.possible_parents[(x, y)],
                                                             (self.lambda_cen, self.lambda_ang,
                                                              self.lambda_siz)),
                               self.pairs_col1, self.pairs_col2)))

    def compute_cost(self):
        """
        This function evaluate the children cost + parent cost
        :return: float - cost value
        """
        overlap_cost = hmn_overlap(self.pairs_temp)
        rank = self.rank_cost()
        pairs_cost = np.array(list(map(self.pair_values, self.pairs_col1, self.pairs_col2)))
        par_cost = self.total_parent_cost()

        return np.sum(pairs_cost[:, 0:2]) + self.lambda_gap * np.sum(pairs_cost[:, 2]) + \
               self.lambda_rat * np.sum(pairs_cost[:, 3]) + self.lambda_over * overlap_cost + \
               self.lambda_rnk * rank + par_cost

    def switch_pairs(self, id, tp):
        """
        This function switch pair tp in the self.pairs_temp, self.col1 and self.col2
        :param id: integer / index of the pair in self.pairs_temp
        :param tp: pair [c_1, c_2]/ that we wanna switch in self.pairs
        :return: switch in pairs_temp, self.col1 and self.col2
        """
        del self.pairs_temp[id]
        self.pairs_temp.insert(id, tp)
        del self.pairs_col1[id]
        self.pairs_col1.insert(id, tp[0])
        del self.pairs_col2[id]
        self.pairs_col2.insert(id, tp[1])

    def boltzmann_machine_optimization(self):
        """
        This finds the best registration to minimize the cost function
        """
        if self.split_num == 0:
            return
        self.initialize_without_neighbor()
        self.cost.append(self.compute_cost())
        self.epoch_cost.append(self.cost[0])
        self.pairs_temp = self.pairs.copy()
        for _ in tqdm(range(self.number_of_epoch)):
            self.overlaps.append(hmn_overlap(self.pairs))
            pairs_cost = np.array(list(map(self.pair_values, self.pairs_col1, self.pairs_col2)))
            self.ratio_cost.append(np.sum(pairs_cost[:, 0:2]))
            self.distance_cost.append(np.sum(pairs_cost[:, 2]))
            self.length_ratio_cost.append(np.sum(pairs_cost[:, 3]))
            self.len_cost.append(self.rank_cost())
            for idx, pair in enumerate(self.pairs):

                p_case = self.all_pairs.copy()
                for p in self.pairs:
                    p_case.remove(p)
                for p in self.pairs_temp:
                    try:
                        p_case.remove(p)
                    except Exception:
                        pass

                if len(p_case) == 0:
                    break

                temp_pair_idx = np.random.choice(len(p_case))
                self.switch_pairs(idx, p_case[temp_pair_idx])

                new_cost = self.compute_cost()
                self.new_cost.append(new_cost)

                delta_cost = new_cost - self.cost[-1]
                if delta_cost < 0:
                    self.cost.append(new_cost)

                else:
                    u = np.random.uniform()
                    if u < np.exp(-delta_cost / self.t):
                        self.cost.append(new_cost)
                    else:
                        self.cost.append(self.cost[-1])
                        self.switch_pairs(idx, pair)

            self.pairs = self.pairs_temp.copy()
            self.epoch_cost.append(self.cost[-1])
            self.t *= self.dt
            if len(self.epoch_cost) > 100 and \
                    np.sum(np.abs(np.diff(self.epoch_cost[-30:]))) < self.epsilon * self.epoch_cost[-100]:
                break

        self.parent_finder()

    def accepted_pair(self, temp=True):
        if temp:
            pair = self.pairs_temp
        else:
            pair = self.pairs
        if hmn_overlap(pair) > 0:
            return False
        else:
            return True

    def parent_finder(self):
        for pair in self.pairs:
            self.pairs_with_parent[self.possible_parents[pair]] = pair


def kinetic(cent0, cent1):
    """
    This function find the movement of two centers
    :param cent0: pairs of integers - center of cells
    :param cent1: pairs of integers - center of second cell
    :return:
    """
    w_i = np.linalg.norm((cent0 - cent1))
    return w_i


def distortion(v_i, v_j, g):
    """
    This function finds the distortion of cell length
    :param v_i: (int, int) - vector v_i from J
    :param v_j: (int, int) - vector v_j from J_+
    :param g: growth rate
    :return: distortion of cell length
    """
    mu = np.square(np.log(np.linalg.norm(v_j) / np.linalg.norm(v_i * g)))

    return mu


def rotation(v_i, v_j):
    """
    This function finds the rotation of two vectors
    :param v_i: (int, int) - vector v_i from J
    :param v_j: (int, int) - vector v_j from J_+
    :return: measure the rotation change
    """
    if np.abs(np.dot(v_i, v_j) / (np.linalg.norm(v_j) * np.linalg.norm(v_i))) > 1:
        mu = 0
    else:
        mu = np.min((np.abs(np.arccos(np.dot(v_i, v_j) / (np.linalg.norm(v_j) * np.linalg.norm(v_i)))),
                     np.abs(np.arccos(np.dot(-v_i, v_j) / (np.linalg.norm(v_j) * np.linalg.norm(v_i))))))

    return mu


def find_neighbors(pindex, triang):
    """
    This function will find neighbors of point pindex from triang and return index of neighbors
    :param pindex: list of integers - index of cell
    :param triang: Delaunay triangulation between the cells
    :return: index of neighbors
    """
    #
    #
    return triang.vertex_neighbor_vertices[1][
           triang.vertex_neighbor_vertices[0][pindex]:
           triang.vertex_neighbor_vertices[0][pindex + 1]]


def end_finder(cent, img, vec):
    """
    This function finds the end points of a cell. We start from the center of the point and move in
    the direction of the vec to the last point of the bacteria cell.

    NOTICE: If the cell is not convex we may have some problem.

    :param cent: (int, int) - center of the cell
    :param img: ndarray - image of the cell
    :param vec: direction to find the endpoint
    :return: to end point of the cell
    """
    if img[cent] != 1:
        print('center is not in the cell, look at function end_finder for definition of Frame')
        return

    end1 = cent
    i = 1
    while True:
        temp = (cent + i * vec).astype(np.uint16)
        if img[temp[0], temp[1]] != 1:
            break
        end1 = temp
        i += 1

    end2 = cent
    i = 1
    while True:
        temp = (cent - i * vec).astype(np.uint16)
        if img[temp[0], temp[1]] != 1:
            break
        end2 = temp
        i += 1

    return np.array((int(end1[1]), int(end1[0]))), np.array((int(end2[1]), int(end2[0])))


def small_cell_finder(frame, hmn):
    """
    This function looks in the frame and find the 'hmn' of shortes cells in the frame.
    :param frame: one of the frame of the movie.
    :param hmn: how many long cell you want. integer
    :return: returns list of 'hmn' long cells in the list.
    """
    cell_length_list = [(x, frame.cells[x].cell_length) for x in frame.cells.keys()]

    length_sorted = sorted(cell_length_list, key=itemgetter(1))

    return length_sorted[:hmn]


def hmn_overlap(pair):
    _, occurrence = np.unique(pair, return_counts=True)
    return sum(occurrence) - len(occurrence)


def major_vote_res(res_file):
    """
    This function return the majority vote of multiple registration mapping (prediction) dictionary
    :param res_file: list of dictionary of registration after boltzmann machine optimization.
    :return: dictionary - One registration which each elements of that is the the majority of
    the all registrations in the res_file.
    """
    tem_dic = {}
    for i in range(1, len(res_file[0]) + 1):
        tem_dic[i] = []
    for j in range(len(res_file)):
        for i in range(1, len(res_file[0]) + 1):
            tem_dic[i].append(res_file[j][i][0])

    for i in range(1, len(tem_dic) + 1):
        tem_dic[i] = [Counter(tem_dic[i]).most_common()[0][0]]

    return tem_dic


def label_img(img_bin, lab=None, color_=(0, 0, 255)):
    """
    write the label of cell on the img_lab image
    :param img_bin: binary image of cells
    :param color_: color for the labels
    :param lab: labeled image
    :return: image of cells with its label
    """
    img_lab = cv2.cvtColor(img_bin * 255, cv2.COLOR_GRAY2BGR)
    if lab is None:
        labeled_img = label(img_bin, connectivity=1)
    else:
        labeled_img = lab

    # print("Number of cells: ", np.max(labeled_img) + 1)
    for i in range(1, np.max(labeled_img) + 1):
        cord_ = np.where(labeled_img == i)
        y_center = int(np.sum(cord_[0]) / len(cord_[0])) + 1
        x_center = int(np.sum(cord_[1]) / len(cord_[1])) - 1
        cv2.putText(img_lab, str(i), (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX, 3 / 10, color_, 1)
    return img_lab


def static_generate_colors(n):
    """
    find n different color
    :param n: number of the colors, integer
    :return: list of different colors
    """

    rgb_values = []
    r = 75
    g = 140
    b = 210
    step = 256 / n
    for _ in range(n):
        r += step
        g += 43 * step
        b += 11 * step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        # r_hex = hex(r)[2:]
        # g_hex = hex(g)[2:]
        # b_hex = hex(b)[2:]
        # hex_values.append('#' + r_hex + g_hex + b_hex)
        rgb_values.append((r, g, b))
    return rgb_values  # hex_values


if __name__ == "__main__":
    img0 = cv2.imread('Sample/J.png', 0)
    img1 = cv2.imread('Sample/J_plus.png', 0)
    sp = SplitTrack(img0, img1)
    print('Linking mother and children started')
    sp.boltzmann_machine_optimization()
    lab0 = sp.J.labeled_image.copy()
    lab1 = sp.J_plus.labeled_image.copy()

    # Label images to display their name on the image
    img0withlabel = label_img(sp.J.image, lab=sp.J.labeled_image)
    img1withlabel = label_img(sp.J_plus.image, lab=sp.J_plus.labeled_image)

    # Save labeled images
    cv2.imwrite('Sample/J_labeled_cells.png', img0withlabel)
    cv2.imwrite('Sample/J_plus_labeled_cells.png', img1withlabel)

    if sp.number_of_pairs == 0:
        registration_dict = {}
    else:
        registration_dict = sp.pairs_with_parent.copy()
        img0 = sp.J.image.copy()
        img1 = sp.J_plus.image.copy()
        for key, val in sp.pairs_with_parent.items():
            img0[sp.J.labeled_image == key] = 0
            lab0[sp.J.labeled_image == key] = 0
            img1[sp.J_plus.labeled_image == val[0]] = 0
            lab1[sp.J_plus.labeled_image == val[0]] = 0
            img1[sp.J_plus.labeled_image == val[1]] = 0
            lab1[sp.J_plus.labeled_image == val[1]] = 0

    nsp = NoSplitTrack(img0, img1, lab0, lab1, growth_rate=1.05)
    print('No split registration started')
    nsp.boltzmann_machine_optimization()
    for key, val in nsp.registration.items():
        registration_dict[nsp.J.transfer_dict[key]] = nsp.J_plus.transfer_dict[val[0]]

    np.save('Sample/Registration.npy', registration_dict)

    col = static_generate_colors(len(registration_dict))

    img0 = cv2.cvtColor(sp.J.image * 255, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(sp.J_plus.image * 255, cv2.COLOR_GRAY2BGR)

    idx = 0
    for key, val in registration_dict.items():
        img0[sp.J.labeled_image == key] = col[idx]
        if isinstance(val, tuple):
            img1[sp.J_plus.labeled_image == val[0]] = col[idx]
            img1[sp.J_plus.labeled_image == val[1]] = col[idx]
        else:
            img1[sp.J_plus.labeled_image == val] = col[idx]
        idx += 1

    cv2.imwrite('Sample/Colored_J.png', img0)
    cv2.imwrite('Sample/Colored_J_plus.png', img1)




