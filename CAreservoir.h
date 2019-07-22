// Neil Babson
// March 2019
// Celluar Automaton Reservoir

#ifndef CARESERVOIR_H
#define CARESERVOIR_H

int R 		        	= 8;  // # of reservoirs
int I 	        		= 4;  // # of CA iterations
int DIFFUSE_LENGTH      	= 40; // Size of reservoir
const int INPUT_LENGTH 		= 4;  //4
int STATES 	        	= 3; 
int NEIGHBORHOOD 		= 3;
int RULELENGTH 	        	= pow(STATES, NEIGHBORHOOD);
int WIDTH			= DIFFUSE_LENGTH * R;
int READOUT_LENGTH	        = R * DIFFUSE_LENGTH * I;
// For 5-bit memory task
const int DISTRACTOR_PERIOD	= 200;
const int SEQUENCE_LENGTH	= DISTRACTOR_PERIOD + 10;
const int TEST_SETS		= 32;
bool DENSITY_RULE		= false;

const std::vector<int> RULE102 = {0,1,1,0,0,1,1,0};
const std::vector<int> RULE90 = {0,1,0,1,1,0,1,0};
const std::vector<int> RULE60 = {0,0,1,1,1,1,0,0};
const std::vector<int> RULE153 = {1,0,0,1,1,0,0,1};
const std::vector<int> RULE195 = {1,1,0,0,0,0,1,1};
const std::vector<int> RULE150 = {1,0,0,1,0,1,1,0};
const std::vector<int> RULE1 = {0,0,0,0,0,0,0,1};
const std::vector<int> RULE0 = {0,0,0,0,0,0,0,0};
const std::vector<int> RULE30 = {0,0,0,1,1,1,1,0};
const std::vector<int> RULE180 = {1,0,1,1,0,1,0,0};

const std::vector<int> RULE3_3 = {1,1,1,1,1,1,2,2,2,2,0,0,2,2,2,1,1,2,0,2,1,1,0,0,0,2,2};

class CA {
    public:
	CA();
	CA(bool uniform);
	void set_input(std::vector<int> input);
	void set_rule(std::vector<int> rule);
	void load_rule(std::string rule_file);
	void load_two_rules(std::string rule_file);
	void apply_rule(alglib::real_2d_array& training_data, int data_index);
	void apply_two_rules(alglib::real_2d_array& training_data, int data_index);
        void build_5_bit_model(alglib::real_2d_array& training_data,
		std::vector<alglib::linearmodel>& output);
	void train_5_bit(alglib::real_2d_array& training_data, bool uniform);
	int test_5_bit(alglib::real_2d_array& training_data, 
		std::vector<alglib::linearmodel>& output);
	void check_CA(alglib::real_2d_array& training_data);
	void draw_CA(alglib::real_2d_array& training_data);
	void save_CA(alglib::real_2d_array& training_data);
	int build_SVM_model(alglib::real_2d_array& training_data);
	int build_scikit_model(alglib::real_2d_array& training_data);
	void set_5_bit_targets();
        void dec_to_base_N(int num, int base, std::vector<int>& ans); 

    private:
        int mod(int x, int y);
	int base_N_to_dec(int num[], int base, int length);
        void call_SVM_functions(int model, int& incorrect, alglib::real_2d_array training_data);
        void python_regression(int model, int& incorrect, alglib::real_2d_array training_data);

	std::vector<std::vector<int>> 	_map;  // [R][INPUT_LENGTH]
	std::vector<std::vector<int>> 	_cell; // [I + 1][WIDTH]
	std::vector<int> 		_rule;
	std::vector<int> 		_rule2;
	int 				_iter;
	std::vector<std::vector<int>> 	_targets;
};

class NegativeModulusException{};
class IncorrectRuleLengthException{};
class BuildRuleFileRequiresIAtLeast3Exception{};
class NonUniformRuleFileFormatException{};
class Expect2State5NeighborhoodException{};

#endif
