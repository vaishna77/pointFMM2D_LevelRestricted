#include <iostream>
#include "FMM2DTree.hpp"


class userkernel: public kernel {
public:
	#ifdef ONEOVERR
	userkernel() {
		isTrans		=	true;
		isHomog		=	true;
		isLogHomog	=	false;
		alpha		=	-1.0;
	};
	double getInteraction(const pts2D r1, const pts2D r2, double a) {
		double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
		double R	=	sqrt(R2);
		if (R < a) {
			return R/a;
		}
		else {
			return a/R;
		}
	};
	#elif LOGR
	userkernel() {
		isTrans		=	true;
		isHomog		=	false;
		isLogHomog	=	true;
		alpha		=	1.0;
	};
	double getInteraction(const pts2D r1, const pts2D r2, double a) {
		double R2	=	(r1.x-r2.x)*(r1.x-r2.x) + (r1.y-r2.y)*(r1.y-r2.y);
		if (R2 < 1e-10) {
			return 0.0;
		}
		/*else if (R2 < a*a) {
			return 0.5*R2*log(R2)/a/a;
		}*/
		else {
			return 0.5*log(R2);
		}
	};
	#endif
	~userkernel() {};
};


int main(int argc, char* argv[]) {
	//int nLevels		=	atoi(argv[1]);
	int nChebNodes	=	atoi(argv[1]);
	int L			=	atoi(argv[2]);
	int N			=	atoi(argv[3]);

	double start, end;

	start	=	omp_get_wtime();
	userkernel* mykernel		=	new userkernel();
	FMM2DTree<userkernel>* A	=	new FMM2DTree<userkernel>(mykernel, nChebNodes, L);

	string fileName				=	"inputData.txt";
	Eigen::VectorXd charges		=	Eigen::VectorXd::Random(N);
	Eigen::VectorXd x			=	L*Eigen::VectorXd::Random(N);
	Eigen::VectorXd y			=	L*Eigen::VectorXd::Random(N);

	double RMAX 		=	RAND_MAX;
	std::ofstream myfile;
	myfile.open(fileName.c_str(),std::ios::out);
	for (int j=0; j<N; ++j) {
		myfile << charges(j) << "\t" << x(j) << "\t" << y(j) << "\n";
	}
	myfile.close();


	A->set_Standard_Cheb_Nodes();

	A->read_inputs(fileName);
	
	A->createTree();
	A->modifyTree();

	A->assign_list2_Neighbor_Interactions();

	A->assign_list1_list3_Interactions();
	
	//A->assign_list4_Interations();
	
	//A->check3();

	end		=	omp_get_wtime();
	double timeCreateTree	=	(end-start);

	std::cout << std::endl << "Number of particles is: " << A->N << std::endl;
	
	std::cout << std::endl << "Time taken to create the tree is: " << timeCreateTree << std::endl;


	start	=	omp_get_wtime();

	A->assemble_Operators_FMM();
	A->get_Transfer_Matrix();
	A->assign_Leaf_ChebNodes();
	
	end		=	omp_get_wtime();
	double timeAssemble		=	(end-start);
	std::cout << std::endl << "Time taken to assemble the operators is: " << timeAssemble << std::endl;

	
	start	=	omp_get_wtime();

	A->evaluate_multipoles();

	end		=	omp_get_wtime();
	double timeAssignCharges=	(end-start);
	std::cout << std::endl << "Time taken to assemble the charges is: " << timeAssignCharges << std::endl;

	start	=	omp_get_wtime();

	A->evaluate_All_M2M();
	end		=	omp_get_wtime();
	double timeM2M			=	(end-start);
	std::cout << std::endl << "Time taken for multipole to multipole is: " << timeM2M << std::endl;

	start	=	omp_get_wtime();

	A->evaluate_list2();
	A->evaluate_list3();
	A->evaluate_list4();

	end		=	omp_get_wtime();
	double timeM2L			=	(end-start);
	std::cout << std::endl << "Time taken for multipole to local is: " << timeM2L << std::endl;

	start	=	omp_get_wtime();
	A->evaluate_All_L2L();
	end		=	omp_get_wtime();
	double timeL2L			=	(end-start);
	std::cout << std::endl << "Time taken for local to local is: " << timeL2L << std::endl;

	start	=	omp_get_wtime();
	A->evaluate_list1();
	
	end		=	omp_get_wtime();
	double timeLeaf			=	(end-start);
	std::cout << std::endl << "Time taken for self and neighbors at leaf is: " << timeLeaf << std::endl;

	double totalTime	=	timeCreateTree+timeAssemble+timeAssignCharges+timeM2M+timeM2L+timeL2L+timeLeaf;
	
	double applyTime	=	timeM2M+timeM2L+timeL2L+timeLeaf;

	std::cout << std::endl << "Total time taken is: " << totalTime << std::endl;

	std::cout << std::endl << "Apply time taken is: " << applyTime << std::endl;

	std::cout << std::endl << "Total Speed in particles per second is: " << A->N/totalTime << std::endl;

	std::cout << std::endl << "Apply Speed in particles per second is: " << A->N/applyTime << std::endl;

	std::cout << std::endl << "Number of particles is: " << A->N << std::endl;


	std::cout << std::endl << "Performing Error check..." << std::endl;

	/*srand(time(NULL));
	int nBox	=	rand()%A->nBoxesPerLevel[nLevels];
	std::cout << std::endl << "Box number is: j: 2; k: 1" << std::endl;
	int j=2, k=1;
	std::cout << std::endl << "Box center is: (" << A->tree[j][k].center.x << ", " << A->tree[j][k].center.y << ");" << std::endl;*/
	//std::cout << std::endl << "Error is: " << A->perform_Error_Check() << std::endl;

	start	=	omp_get_wtime();

	A->perform_Error_Check();
	
	end		=	omp_get_wtime();
	double errorTime	=	(end-start);

	std::cout << std::endl << "Time taken to compute error is: " << errorTime << std::endl;
	
	std::cout << std::endl;
	//A->check4();

}
