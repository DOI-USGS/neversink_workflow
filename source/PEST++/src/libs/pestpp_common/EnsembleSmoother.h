#ifndef ENSEMBLESMOOTHER_H_
#define ENSEMBLESMOOTHER_H_

#include <map>
#include <random>
#include <mutex>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "FileManager.h"
#include "ObjectiveFunc.h"
#include "OutputFileWriter.h"
#include "PerformanceLog.h"
#include "RunStorage.h"
#include "covariance.h"
#include "RunManagerAbstract.h"
#include "ObjectiveFunc.h"
#include "Localizer.h"
#include "EnsembleMethodUtils.h"



class LocalUpgradeThread
{
public:

	LocalUpgradeThread(PerformanceLog *_performance_log, unordered_map<string, Eigen::VectorXd> &_par_resid_map, unordered_map<string, Eigen::VectorXd> &_par_diff_map,
		unordered_map<string, Eigen::VectorXd> &_obs_resid_map, unordered_map<string, Eigen::VectorXd> &_obs_diff_map,
		Localizer &_localizer, unordered_map<string, double> &_parcov_inv_map,
		unordered_map<string, double> &_weight_map, ParameterEnsemble &_pe_upgrade,
		unordered_map<string, pair<vector<string>, vector<string>>> &_cases,
		unordered_map<string, Eigen::VectorXd> &_Am_map, Localizer::How &_how);

	//Eigen::DiagonalMatrix<double, Eigen::Dynamic> get_matrix_from_map(vector<string> &names, map<string, double> &dmap);	
	//Eigen::MatrixXd get_matrix_from_map(int num_reals, vector<string> &names, map<string, Eigen::VectorXd> &emap);


	void work(int thread_id, int iter, double cur_lam);


private:
	PerformanceLog * performance_log;
	Localizer::How how;
	vector<string> keys;
	int count, total;
	//double eigthresh, cur_lam;
	//int maxsing, num_reals,iter, thread_id;
	//bool use_approx, use_prior_scaling;

	unordered_map<string, pair<vector<string>, vector<string>>> &cases;

	ParameterEnsemble &pe_upgrade;
	//PhiHandler &ph;
	Localizer &localizer;
	unordered_map<string, double> &parcov_inv_map;
	unordered_map<string, double> &weight_map;

	unordered_map<string, Eigen::VectorXd> &par_resid_map, &par_diff_map, &Am_map;
	unordered_map<string, Eigen::VectorXd> &obs_resid_map, &obs_diff_map;

	mutex ctrl_lock, weight_lock, loc_lock, parcov_lock;
	mutex obs_resid_lock, obs_diff_lock, par_resid_lock;
	mutex par_diff_lock, am_lock, put_lock;
	mutex next_lock;
	
};


class IterEnsembleSmoother
{
	
public:
	IterEnsembleSmoother(Pest& _pest_scenario, FileManager& _file_manager,
		OutputFileWriter& _output_file_writer, PerformanceLog* _performance_log,
		RunManagerAbstract* _run_mgr_ptr);
	
	void initialize();
	void iterate_2_solution();
	void finalize();
	void throw_ies_error(string message);
	bool should_terminate();
	

private:
	int  verbose_level;
	Pest &pest_scenario;
	FileManager &file_manager;
	std::mt19937 rand_gen;
	std::mt19937 subset_rand_gen;
	OutputFileWriter &output_file_writer;
	PerformanceLog *performance_log;
	RunManagerAbstract* run_mgr_ptr;
	L2PhiHandler ph;
	ParChangeSummarizer pcs;
	Covariance parcov, obscov;
	double reg_factor;

	

	bool use_localizer;
	Localizer localizer;

	int num_threads;

	set<string> pp_args;

	int iter,subset_size;
	bool use_subset;

	double last_best_lam, last_best_mean,last_best_std;
	vector<double> best_mean_phis;
	double best_phi_yet;

	int consec_bad_lambda_cycles;

	double lambda_max, lambda_min;
	int warn_min_reals, error_min_reals;
	vector<double> lam_mults;
	
	vector<string> oe_org_real_names, pe_org_real_names;
	vector<string> act_obs_names, act_par_names;
	vector<int> subset_idxs;

	ParameterEnsemble pe, pe_base;
	ObservationEnsemble oe, oe_base;
	//Eigen::MatrixXd prior_pe_diff;
	//Eigen::MatrixXd Am;
	Eigen::DiagonalMatrix<double,Eigen::Dynamic> obscov_inv_sqrt, parcov_inv_sqrt;

	bool oe_drawn, pe_drawn;

	//bool solve_old();
	bool solve_new();

	ParameterEnsemble calc_localized_upgrade_threaded(double cur_lam, unordered_map<string, pair<vector<string>, vector<string>>> &loc_map);

	vector<int> run_ensemble(ParameterEnsemble &_pe, ObservationEnsemble &_oe, const vector<int> &real_idxs=vector<int>());
	vector<ObservationEnsemble> run_lambda_ensembles(vector<ParameterEnsemble> &pe_lams, vector<double> &lam_vals, vector<double> &scale_vals);
	
	void report_and_save();
	void save_mat(string prefix, Eigen::MatrixXd &mat);
	bool initialize_pe(Covariance &cov);
	bool initialize_oe(Covariance &cov);
	void initialize_restart();
	void initialize_parcov();
	void initialize_obscov();
	void drop_bad_phi(ParameterEnsemble &_pe, ObservationEnsemble &_oe, bool is_subset=false);
	template<typename T, typename A>
	void message(int level, const string &_message, vector<T, A> _extras, bool echo=true);
	void message(int level, const string &_message);

	template<typename T>
	void message(int level, const string &_message, T extra);

	void sanity_checks();

	void add_bases();

	void update_reals_by_phi(ParameterEnsemble &_pe, ObservationEnsemble &_oe);

	vector<string> detect_prior_data_conflict();

	void set_subset_idx(int size);
	Eigen::MatrixXd get_Am(const vector<string> &real_names, const vector<string> &par_names);

	void zero_weight_obs(vector<string>& obs_to_zero_weight, bool update_obscov=true,bool update_oe_base=true);

	void norm_map_report(map<string, double>& norm_map, string tag, double thres = 0.1);

};

#endif
