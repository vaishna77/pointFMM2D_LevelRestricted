#ifndef _FMM2DTree_HPP__
#define _FMM2DTree_HPP__

#include <vector>
#include <Eigen/Dense>
#include <fstream>

#define EIGEN_DONT_PARALLELIZE
#include <iostream>
#include <cmath>
#include <stdlib.h>     /* srand, rand */
#include <time.h>   

using namespace std;

const double PI	=	3.1415926535897932384;
const int MIN_CHARGES_PER_BOX = 15;
struct pts2D {
	double x,y;
};

struct charge {
	double q,x,y;
};

struct level_box {
	int level,box;
};

class FMM2DBox {
public:
	bool exists;
	int boxNumber;
	int parentNumber;
	int childrenNumbers[4];
	int neighborNumbers[8];
	int innerNumbers[16];
	int outerNumbers[24];

	int list1_smallNumbers[12];	//	fine neighbors
	int list1_largeNumbers[12]; //	coarse neighbors
	int list3Numbers[20];	//	seperated fine neighbors
	int list4Numbers[20];	//	coarse level interaction list or seperated coarse boxes

	FMM2DBox () {
		boxNumber		=	-1;
		parentNumber	=	-1;
		for (int l=0; l<4; ++l) {
			childrenNumbers[l]	=	-1;
		}
		for (int l=0; l<8; ++l) {
			neighborNumbers[l]	=	-1;
		}
		for (int l=0; l<16; ++l) {
			innerNumbers[l]		=	-1;
		}
		for (int l=0; l<24; ++l) {
			outerNumbers[l]		=	-1;
		}
		for (int l=0; l<12; ++l) {
			list1_smallNumbers[l]	=	-1;
		}
		for (int l=0; l<12; ++l) {
			list1_largeNumbers[l]	=	-1;
		}
		for (int l=0; l<20; ++l) {
			list3Numbers[l]		=	-1;
		}
		for (int l=0; l<20; ++l) {
			list4Numbers[l]		=	-1;
		}		
	}
	//	charge database of charges present in the box
	std::vector<int> charge_indices;
  	
	Eigen::VectorXd multipoles;
	Eigen::VectorXd locals;

	pts2D center;

	//	The following will be stored only at the leaf nodes
	std::vector<pts2D> chebNodes;
};

class kernel {
public:
	bool isTrans;		//	Checks if the kernel is translation invariant, i.e., the kernel is K(r).
	bool isHomog;		//	Checks if the kernel is homogeneous, i.e., K(r) = r^{alpha}.
	bool isLogHomog;	//	Checks if the kernel is log-homogeneous, i.e., K(r) = log(r^{alpha}).
	double alpha;		//	Degree of homogeneity of the kernel.
	kernel() {};
	~kernel() {};
	virtual double getInteraction(const pts2D r1, const pts2D r2, double a){
		return 0.0;
	};	//	Kernel entry generator
};

template <typename kerneltype>
class FMM2DTree {
public:
	kerneltype* K;
	int nLevels;			//	Number of levels in the tree.
	int nChebNodes;			//	Number of Chebyshev nodes along one direction.
	int rank;				//	Rank of interaction, i.e., rank = nChebNodes*nChebNodes.
	int N;					//	Number of particles.
	double L;				//	Semi-length of the simulation box.
	double smallestBoxSize;	//	This is L/2.0^(nLevels).
	double a;				//	Cut-off for self-interaction. This is less than the length of the smallest box size.

	std::vector<int> nBoxesPerLevel;			//	Number of boxes at each level in the tree.
	std::vector<double> boxRadius;				//	Box radius at each level in the tree assuming the box at the root is [-1,1]^2
	std::vector<double> boxHomogRadius;			//	Stores the value of boxRadius^{alpha}
	std::vector<double> boxLogHomogRadius;		//	Stores the value of alpha*log(boxRadius)
	std::vector<std::vector<FMM2DBox> > tree;	//	The tree storing all the information.
	
	//	childless_boxes
	std::vector<level_box> childless_boxes;

	//	charge in coloumbs and its location
	std::vector<charge> charge_database;
  	
	//	Chebyshev nodes
	std::vector<double> standardChebNodes1D;
	std::vector<pts2D> standardChebNodes;
	std::vector<pts2D> standardChebNodesChild;
	std::vector<pts2D> leafChebNodes;

	//	Different Operators
	Eigen::MatrixXd selfInteraction;		//	Needed only at the leaf level.
	Eigen::MatrixXd neighborInteraction[8];	//	Neighbor interaction only needed at the leaf level.
	Eigen::MatrixXd M2M[4];					//	Transfer from multipoles of 4 children to multipoles of parent.
	Eigen::MatrixXd L2L[4];					//	Transfer from locals of parent to locals of 4 children.
	Eigen::MatrixXd M2LInner[16];			//	M2L of inner interactions. This is done on the box [-L,L]^2.
	Eigen::MatrixXd M2LOuter[24];			//	M2L of outer interactions. This is done on the box [-L,L]^2.

	Eigen::MatrixXd List1_Small_Interaction[12];
	Eigen::MatrixXd List1_Large_Interaction[12];
	Eigen::MatrixXd List3_Interaction[20];
	Eigen::MatrixXd List4_Interaction[20];

// public:
	FMM2DTree(kerneltype* K, /*int nLevels,*/ int nChebNodes, double L) {
		this->K					=	K;
		//this->nLevels			=	nLevels;
		this->nChebNodes		=	nChebNodes;
		this->rank				=	nChebNodes*nChebNodes;
		this->L					=	L;
	}

	std::vector<pts2D> shift_Cheb_Nodes(double xShift, double yShift) {
		std::vector<pts2D> shiftedChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	standardChebNodes[k].x+2*xShift;
			temp.y	=	standardChebNodes[k].y+2*yShift;
			shiftedChebNodes.push_back(temp);
		}
		return shiftedChebNodes;
	}
	
	std::vector<pts2D> shift_Leaf_Cheb_Nodes(double xShift, double yShift) {
		std::vector<pts2D> shiftedChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	leafChebNodes[k].x+xShift;
			temp.y	=	leafChebNodes[k].y+yShift;
			shiftedChebNodes.push_back(temp);
		}
		return shiftedChebNodes;
	}


	//	shifted_scaled_cheb_nodes	//	used in evaluating multipoles
	std::vector<pts2D> shift_scale_Cheb_Nodes(double xShift, double yShift, double radius) {
		std::vector<pts2D> shifted_scaled_ChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	radius*standardChebNodes[k].x+xShift;
			temp.y	=	radius*standardChebNodes[k].y+yShift;
			shifted_scaled_ChebNodes.push_back(temp);
		}
		return shifted_scaled_ChebNodes;
	}


	//	get_ChebPoly
	double get_ChebPoly(double x, int n) {
		return cos(n*acos(x));
	}

	//	get_S
	double get_S(double x, double y, int n) {
		double S	=	0.5;
		for (int k=1; k<n; ++k) {
			S+=get_ChebPoly(x,k)*get_ChebPoly(y,k);
		}
		return 2.0/n*S;
	}
	//	set_Standard_Cheb_Nodes
	void set_Standard_Cheb_Nodes() {
		for (int k=0; k<nChebNodes; ++k) {
			standardChebNodes1D.push_back(-cos((k+0.5)/nChebNodes*PI));
		}
		pts2D temp1;
		for (int j=0; j<nChebNodes; ++j) {
			for (int k=0; k<nChebNodes; ++k) {
				temp1.x	=	standardChebNodes1D[k];
				temp1.y	=	standardChebNodes1D[j];
				standardChebNodes.push_back(temp1);
			}
		}
		//	Left Bottom child, i.e., Child 0
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Bottom child, i.e., Child 1
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Top child, i.e., Child 2
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Left Top child, i.e., Child 3
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
	}

	void get_Transfer_Matrix() {
		for (int l=0; l<4; ++l) {
			L2L[l]	=	Eigen::MatrixXd(rank,rank);
			for (int j=0; j<rank; ++j) {
				for (int k=0; k<rank; ++k) {
					L2L[l](j,k)	=	get_S(standardChebNodes[k].x, standardChebNodesChild[j+l*rank].x, nChebNodes)*get_S(standardChebNodes[k].y, standardChebNodesChild[j+l*rank].y, nChebNodes);
				}
			}
		}
		for (int l=0; l<4; ++l) {
			M2M[l]	=	L2L[l].transpose();
		}
	}


	

	void read_inputs(string fileName) {
	//reading charge data from a file
		ifstream myfile;
		myfile.open (fileName.c_str());
		charge a;
  		double b;
  		int c =0;
  		while (myfile >> b){
			if(c==0){
				a.q	=	b;
				c = 1;
			}
			else if(c==1){
				a.x	=	b;
				c = 2;
			}
			else{
				a.y	=	b;
				c=0;
				charge_database.push_back(a);
			}
	        }
  		myfile.close();
		
	}

	void createTree() {
		//	First create root and add to tree
		FMM2DBox root;
		root.exists	=	true;
		root.boxNumber		=	0;
		root.parentNumber	=	-1;
		//not sure if it has children
		/*
		#pragma omp parallel for
		for (int l=0; l<4; ++l) {
			root.childrenNumbers[l]	=	l;
		}
		*/
		#pragma omp parallel for
		for (int l=0; l<8; ++l) {
			root.neighborNumbers[l]	=	-1;
		}
		#pragma omp parallel for
		for (int l=0; l<16; ++l) {
			root.innerNumbers[l]	=	-1;
		}
		#pragma omp parallel for
		for (int l=0; l<24; ++l) {
			root.outerNumbers[l]	=	-1;
		}
		for (int j=0; j<charge_database.size(); j++) {
			root.charge_indices.push_back(j);
		}
		root.center.x = 0.0;
		root.center.y = 0.0;
		
		std::vector<FMM2DBox> rootLevel;
		rootLevel.push_back(root);
		tree.push_back(rootLevel);
		
		nBoxesPerLevel.push_back(1);
		boxRadius.push_back(L);
		boxHomogRadius.push_back(pow(L,K->alpha));
		boxLogHomogRadius.push_back(K->alpha*log(L));

		int j=1;
		while (1) {
			nBoxesPerLevel.push_back(4*nBoxesPerLevel[j-1]);
			boxRadius.push_back(0.5*boxRadius[j-1]);
			boxHomogRadius.push_back(pow(0.5,K->alpha)*boxHomogRadius[j-1]);
			boxLogHomogRadius.push_back(boxLogHomogRadius[j-1]-K->alpha*log(2));

			std::vector<FMM2DBox> level_vb;
			bool stop_refining = true;
			for (int k=0; k<nBoxesPerLevel[j-1]; ++k) { // no. of boxes in parent level
				/////////////////////////////////////////////////////////////////////////////////////////////
				// check if there are atleast MIN_CHARGES_PER_BOX charges inside the box to make it have children
				int s = 0;
				//	check for the charges present in parent
				if (tree[j-1][k].exists) { // if the parent is non-existent it's meaningless to have  children for such a box. so avoid checking
					for (int i=0; i<tree[j-1][k].charge_indices.size(); ++i) {
						
						if (charge_database[tree[j-1][k].charge_indices[i]].x <= (tree[j-1][k].center.x + boxRadius[j-1]) && charge_database[tree[j-1][k].charge_indices[i]].x > (tree[j-1][k].center.x - boxRadius[j-1]) && charge_database[tree[j-1][k].charge_indices[i]].y <= (tree[j-1][k].center.y + boxRadius[j-1]) && charge_database[tree[j-1][k].charge_indices[i]].y > (tree[j-1][k].center.y - boxRadius[j-1])) {
							++s;
							if(s == MIN_CHARGES_PER_BOX) {
								break;
							}
						}
					}
					
				}
				// if a box can have children it has to be 4
				
				for (int l=0; l<4; ++l) {
					FMM2DBox box;
					if (l==0) {
						box.center.x = tree[j-1][k].center.x - boxRadius[j];
						box.center.y = tree[j-1][k].center.y - boxRadius[j];
					}
					else if (l==1) {
						box.center.x = tree[j-1][k].center.x + boxRadius[j];
						box.center.y = tree[j-1][k].center.y - boxRadius[j];
					}
					else if (l==2) {
						box.center.x = tree[j-1][k].center.x + boxRadius[j];
						box.center.y = tree[j-1][k].center.y + boxRadius[j];
					}
					else {
						box.center.x = tree[j-1][k].center.x - boxRadius[j];
						box.center.y = tree[j-1][k].center.y + boxRadius[j];
					}
					// writing childrenNumbers, boxNumber, parentNumber irrespective of whether they exist or not because it helps later in forming the 4 lists					
					tree[j-1][k].childrenNumbers[l]	=	4*k+l; 
				 	box.boxNumber		=	4*k+l;
					box.parentNumber	=	k;

					if(s == MIN_CHARGES_PER_BOX) { // can have 4 children. and these
						
						box.exists		=	true;
						stop_refining = false;
						// parent cannot have children if there are not atleast enough charges in it
						// in this case parent can have children, so create the child boxes and add them to the tree
						
						// distribution of charges among the children
						// if charge is on right and top boundary of box it belongs to that box
						//cout << "center x: " << box.center.x << "	center y: " << box.center.y << "			radius: " << boxRadius[j] << endl;
						for (int i=0; i<tree[j-1][k].charge_indices.size(); ++i) {
							
							
							if (charge_database[tree[j-1][k].charge_indices[i]].x <= (box.center.x + boxRadius[j]) && charge_database[tree[j-1][k].charge_indices[i]].x > (box.center.x - boxRadius[j]) && charge_database[tree[j-1][k].charge_indices[i]].y <= (box.center.y + boxRadius[j]) && charge_database[tree[j-1][k].charge_indices[i]].y > (box.center.y - boxRadius[j])) {
								box.charge_indices.push_back(tree[j-1][k].charge_indices[i]);	
							
							}
						}
						//level.push_back(box); // putting a box into tree only if the parent box has atleast MIN_CHARGES_PER_BOX charges because a box is a box only if it's parent has atleast MIN_CHARGES_PER_BOX charges
					}	
					else {
						box.exists		=	false;
						if (l == 0) { // writing into childless boxes only once
							level_box lb;
							lb.level = j-1;
							lb.box = k;
							if (tree[j-1][k].exists) {
								childless_boxes.push_back(lb);
							}
						}
						// meaningless to have parentNumber to a non-existing child
					}
					level_vb.push_back(box); // putting non existent boxes also into tree
				}
			}
			if (stop_refining) 
				break;
			else {
				tree.push_back(level_vb);
				++j;
			}
			
		}
		nLevels = j-1;
		cout << "nLevels: " << nLevels << endl;
		smallestBoxSize	=	boxRadius[nLevels];
		a		=	smallestBoxSize;
		N		=	rank*childless_boxes.size();
		
		std::vector<FMM2DBox> level_vb;
		for (int k=0; k<4*nBoxesPerLevel[nLevels]; ++k) {
			FMM2DBox box;
			box.exists = false;
			level_vb.push_back(box);
		}
		tree.push_back(level_vb);
	}






////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//	Assigns the interactions for child0 of a box
	void assign_Child0_Interaction0(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k;
		int nN, nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[3]	=	nC+1;
			tree[nL][nC].neighborNumbers[4]	=	nC+2;
			tree[nL][nC].neighborNumbers[5]	=	nC+3;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*	 _____________|______|  */
			/*	|	   |	  |			*/
			/*	|  I15 |  N0  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 ) {
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______			  		*/
			/*	|	   |	  			*/
			/*	|  **  |				*/
			/*	|______|______			*/
			/*	|	   |	  |			*/
			/*	|  N1  |  N2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1 ) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[3];
				
			}
		}

		
		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  |	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **  |	*/
			/*	|______|______|______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[2];
			}
		}
	}

	//	Assigns the interactions for child1 of a box
	void assign_Child1_Interaction0(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+1;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[7]	=	nC-1;
			tree[nL][nC].neighborNumbers[5]	=	nC+1;
			tree[nL][nC].neighborNumbers[6]	=	nC+2;
		}

		

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 ______|______|			*/
			/*	|	   |	  |			*/
			/*	|  N0  |  N1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[3];
												
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |	 I7	 |  */
			/*	 ______|______|______|	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |  I6  |  */
			/*	|______|______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[3];	
											
			}
		}
	}

	//	Assigns the interactions for child2 of a box
	void assign_Child2_Interaction0(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+2;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N0  |  N1  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[0]	=	nC-2;
			tree[nL][nC].neighborNumbers[1]	=	nC-1;
			tree[nL][nC].neighborNumbers[7]	=	nC+1;
		}

	

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*	 ____________________	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |	 I6	 |  */
			/*	|______|______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |  */
			/*		   |______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[3];		
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |  I7  |	*/
			/*	 ______|______|______|	*/
			/*	|	   |	  			*/
			/*	|  **  |	  			*/
			/*	|______|	  			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				
			}
		}
	}

	//	Assigns the interactions for child3 of a box
	void assign_Child3_Interaction0(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+3;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N1  |  N2  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[1]	=	nC-3;
			tree[nL][nC].neighborNumbers[2]	=	nC-2;
			tree[nL][nC].neighborNumbers[3]	=	nC-1;
		}



		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[1];
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[1];
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 ____________________	*/
			/*	|	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **	 |	*/
			/*	|______|______|______|	*/
			/*  |	   |	  |		 	*/
			/*	|  I15 |  N0  |	 	 	*/
			/*	|______|______|		 	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[2];
			}
		}
	}


	//	Assigns the interactions for the children of a box
	void assign_Box_Interactions0(int j, int k) {
		assign_Child0_Interaction0(j,k);
		assign_Child1_Interaction0(j,k);
		assign_Child2_Interaction0(j,k);
		assign_Child3_Interaction0(j,k);
	}

	//	Assigns the interactions for the children all boxes at a given level
	void assign_Level_Interactions0(int j) {
		for (int k=0; k<tree[j].size(); ++k) {
			if (tree[j+1][4*k].exists) { // do this only if the box is a parent
				assign_Box_Interactions0(j,tree[j][k].boxNumber);
			}
		}
	}

	//	Assigns the interactions for the children of all boxes in the tree
	//	Assigns colleagues(neighbors) and list2 (inner and outer numbers) needed for M2L(same size)
	void assign_list2_Neighbor_Interactions0() {
		for (int j=0; j<nLevels; ++j) {
			assign_Level_Interactions0(j);
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	void modifyTree() {
		assign_list2_Neighbor_Interactions0();
		bool unmodified = true;
		for (int l=0; l<childless_boxes.size(); ++l) {
			int j = childless_boxes[l].level;
			int k = childless_boxes[l].box;
			if (tree[j][k].exists) {
				for (int i=0; i<8; ++i) {
					int nN = tree[j][k].neighborNumbers[i];
					if (nN != -1) {
						int c = j-1;
						int d = nN/4;
						while (!tree[c][d].exists) {
							unmodified  = false;
							int pN = d/4;
							//	create all its children
							tree[c][4*pN].exists = true;
							tree[c][4*pN+1].exists = true;
							tree[c][4*pN+2].exists = true;
							tree[c][4*pN+3].exists = true;
							--c;
							d= pN;
						}
						if (c != j-1) {
							assign_charges(c,d);
						}
					}
				}
			}
		}
		if (unmodified  == false) {
			childless_boxes.clear();
			assign_childless_boxes();
			modifyTree();
		}
	}



	void assign_charges(int c, int d) {
		if (tree[c][d].exists && tree[c+1][4*d].exists) {
			for (int e=0; e<4; ++e) {
				for (int i=0; i<tree[c][d].charge_indices.size(); ++i) {
					if (charge_database[tree[c][d].charge_indices[i]].x <= (tree[c+1][4*d+e].center.x + boxRadius[c+1]) && charge_database[tree[c][d].charge_indices[i]].x > (tree[c+1][4*d+e].center.x - boxRadius[c+1]) && charge_database[tree[c][d].charge_indices[i]].y <= (tree[c+1][4*d+e].center.y + boxRadius[c+1]) && charge_database[tree[c][d].charge_indices[i]].y > (tree[c+1][4*d+e].center.y - boxRadius[c+1])) {
					tree[c+1][4*d+e].charge_indices.push_back(tree[c][d].charge_indices[i]);	
					}
				}
			}
			assign_charges(c+1,4*d);
			assign_charges(c+1,4*d+1);
			assign_charges(c+1,4*d+2);
			assign_charges(c+1,4*d+3);
		}
	}




	void assign_childless_boxes() {
		for (int j=0; j<=nLevels; ++j) {
			for (int k=0; k<tree[j].size(); ++k) {
				if (tree[j][k].exists) {
					int cN = tree[j][k].childrenNumbers[0];
					if (tree[j+1][cN].exists) {
					}
					else {
						level_box lb;
						lb.level = j;
						lb.box = k;
						childless_boxes.push_back(lb);
					}
				}
			}
		}
	}








	//	Assigns the interactions for child0 of a box
	void assign_Child0_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k;
		int nN, nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[3]	=	nC+1;
			tree[nL][nC].neighborNumbers[4]	=	nC+2;
			tree[nL][nC].neighborNumbers[5]	=	nC+3;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*	 _____________|______|  */
			/*	|	   |	  |			*/
			/*	|  I15 |  N0  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 ) {
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______			  		*/
			/*	|	   |	  			*/
			/*	|  **  |				*/
			/*	|______|______			*/
			/*	|	   |	  |			*/
			/*	|  N1  |  N2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1 ) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[3];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				}
			}
		}

		//	Assign children of parent's second neighbor
		{
			/************************************/
			/*	 ______			  				*/
			/*	|	   |	  					*/
			/*	|  **  |						*/
			/*	|______|	   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I5  |  O8  |	*/
			/*				  |______|______|	*/
			/*				  |	     |	    |	*/
			/*				  |  I4  |  O7  |	*/
			/*				  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[8]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I7  |  O10 |	*/
			/*	 ______		  |______|______|	*/
			/*	|	   |	  |	     |	    |	*/
			/*	|  **  |	  |  I6  |  O9  |	*/
			/*	|______|	  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[9]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[10]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[3];
			}

		}

		//	Assign children of parent's fourth neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  O13 |  O12 |	*/
			/*				  |______|______|	*/
			/*				  |	     |	    |	*/
			/*		    	  |  I8  |  O11 |	*/
			/*		    	  |______|______|	*/
			/*									*/
			/*									*/
			/*	 ______							*/
			/*  |      |						*/
			/*  |  **  |						*/
			/*  |______|						*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[11]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[12]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[13]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O15 |  O14 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*	 ______				*/
			/*  |	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[14]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[15]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  O17 |  O16 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*							*/
			/*							*/
			/*				   ______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[16]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[17]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  |	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **  |	*/
			/*	|______|______|______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[2];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}
	}

	//	Assigns the interactions for child1 of a box
	void assign_Child1_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+1;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[7]	=	nC-1;
			tree[nL][nC].neighborNumbers[5]	=	nC+1;
			tree[nL][nC].neighborNumbers[6]	=	nC+2;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/************************************/
			/*				   		  ______	*/
			/*				  	     |		|	*/
			/*				         |	**  |	*/
			/*	 _____________       |______|  	*/
			/*	|	   |	  |					*/
			/*	|  O22 |  I15 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O23 |  I0  |					*/
			/*	|______|______|					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[23]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[22]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 ______|______|			*/
			/*	|	   |	  |			*/
			/*	|  N0  |  N1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[3];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				}								
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[2];
				}
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |	 I7	 |  */
			/*	 ______|______|______|	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |  I6  |  */
			/*	|______|______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[3];	
				if(tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				}							
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  O14 |  O13 |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*				  			*/
			/*				  			*/
			/*	 ______					*/
			/*	|	   |				*/
			/*  |  **  |				*/
			/*  |______|				*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[13]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[14]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O16 |  O15 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*		    ______		*/
			/* 		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[15]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[16]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/************************************/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O18 |  O17 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O19 |  I12 |					*/
			/*	|______|______|					*/
			/*									*/
			/*									*/
			/*				   		  ______	*/
			/*  			  		 |		|	*/
			/*				  		 |	** 	|	*/
			/*				  		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[19]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[17]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[18]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O20 |  I13 |					*/
			/*	|______|______|		  ______	*/
			/*  |	   |	  |		 |		|	*/
			/*	|  O21 |  I14 |	 	 |	**  |	*/
			/*	|______|______|		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[21]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[20]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child2 of a box
	void assign_Child2_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+2;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N0  |  N1  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[0]	=	nC-2;
			tree[nL][nC].neighborNumbers[1]	=	nC-1;
			tree[nL][nC].neighborNumbers[7]	=	nC+1;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/************************************/
			/*				   		  ______	*/
			/*				  	     |		|	*/
			/*				         |	**  |	*/
			/*				         |______|  	*/
			/*									*/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O23 |  I0  |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O0  |  O1  |					*/
			/*	|______|______|					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[23]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 	   |______|			*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O2  |  O3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  O4  |  O5  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*	 ____________________	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |	 I6	 |  */
			/*	|______|______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |  */
			/*		   |______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[3];		
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[2];
				}
						
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |  I7  |	*/
			/*	 ______|______|______|	*/
			/*	|	   |	  			*/
			/*	|  **  |	  			*/
			/*	|______|	  			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[1];
					tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[2];
					tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[2];
					tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/************************************/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O19 |  I12 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O20 |  I13 |					*/
			/*	|______|______|		  ______	*/
			/*  			  		 |		|	*/
			/*				  		 |	** 	|	*/
			/*				  		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[20]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[19]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________		  ______	*/
			/*	|	   |	  |		 |	    |	*/
			/*	|  O21 |  I14 |		 |	**	|	*/
			/*	|______|______|		 |______|	*/
			/*  |	   |	  |		 			*/
			/*	|  O22 |  I15 |	 	 			*/
			/*	|______|______|		 			*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[22]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[21]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child3 of a box
	void assign_Child3_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+3;
		int nN,nNC;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N1  |  N2  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[1]	=	nC-3;
			tree[nL][nC].neighborNumbers[2]	=	nC-2;
			tree[nL][nC].neighborNumbers[3]	=	nC-1;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|  */
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O1  |  O2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |				*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O3  |  O4  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/************************************/
			/*	 ______		  					*/
			/*	|	   |						*/
			/*	|  **  |	  					*/
			/*	|______|						*/
			/*									*/
			/*									*/
			/*				   _____________	*/
			/*		   		  |	     |	    |	*/
			/*		   		  |  I4  |  O7  |	*/
			/*		   		  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*	       		  |  O5  |  O6  |	*/
			/*		   		  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[2];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/************************************/
			/*	 ______		   _____________	*/
			/*	|	   |	  |	     |		|	*/
			/*	|  **  |      |	 I6	 |  O9	|	*/
			/*	|______|	  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I5  |  O8  |  	*/
			/*		   		  |______|______| 	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[3];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[8]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[9]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I8  |  O11 |	*/
			/*		   		  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I7  |  O10 |	*/
			/*	 ______	      |______|______|	*/
			/*	|	   |	  					*/
			/*	|  **  |	  					*/
			/*	|______|	  					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[4];
			nNC	=	4*nN;
			if (nN != -1 && tree[j+1][nNC].exists) {
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[10]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[11]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[2];
					tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[2];
					tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[3];				
				}
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 ____________________	*/
			/*	|	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **	 |	*/
			/*	|______|______|______|	*/
			/*  |	   |	  |		 	*/
			/*	|  I15 |  N0  |	 	 	*/
			/*	|______|______|		 	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			nNC	=	4*nN;
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				if (tree[j+1][nNC].exists) {
					tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[0];
					tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[3];	
				}			
			}
		}
	}

	//	Assigns the interactions for the children of a box
	void assign_Box_Interactions(int j, int k) {
		assign_Child0_Interaction(j,k);
		assign_Child1_Interaction(j,k);
		assign_Child2_Interaction(j,k);
		assign_Child3_Interaction(j,k);
	}

	//	Assigns the interactions for the children all boxes at a given level
	void assign_Level_Interactions(int j) {
		//cout << endl << "tree[j].size: " << tree[j].size() << endl;
		//#pragma omp parallel for
		for (int k=0; k<tree[j].size(); ++k) {
			//cout << k << endl;
			if (tree[j+1][4*k].exists) { // do this only if the box is a parent
				//cout << j+1 << "," << 4*k << "level box" << endl;
				assign_Box_Interactions(j,tree[j][k].boxNumber);
			}
		}
	}

	//	Assigns the interactions for the children of all boxes in the tree
	//	Assigns colleagues(neighbors) and list2 (inner and outer numbers) needed for M2L(same size)
	void assign_list2_Neighbor_Interactions() {
		for (int j=0; j<nLevels; ++j) {
			assign_Level_Interactions(j);
		}
	}


	//	Assigns list1 for childless boxes	
	void assign_list1_list3_Interactions() {
		for (int j=0; j<childless_boxes.size(); ++j) {
			//childless_boxes[j].level = l;
			//childless_boxes[j].box = b;
			assign_list1_list3_box_Interactions(childless_boxes[j]);
		}
	}





	//	LIST 1
	void assign_list1_list3_box_Interactions(const level_box lb) { // for boxes which donot have children
		level_box temp;
		
		//	if box is childless it also is a member of list 1
		//tree[lb.level][lb.box].list1.push_back(lb);
		level_box prev_add, neigh0_add;
		prev_add.level = -1;
		prev_add.box = 0;
		neigh0_add.level = -1;
		neigh0_add.box = 0;
		int j,k;
		
		//	NEIGHBOR 0,2,4,6 CORNERS
		//	NEIGHBOR 1,3,5,7 SIDE BY SIDE

		//	NEIGHBOR 0
		//if (lb.level == 2 && lb.box == 10) {
		k = tree[lb.level][lb.box].neighborNumbers[0]; // neighbor is in same level
		j = lb.level;
		//cout << "j: " << j << "	k: " << k << endl;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			if (tree[j][k].exists) {
				if (tree[j+1][4*k].exists) {	
					tree[lb.level][lb.box].list1_smallNumbers[0] = 4*k+2;

					tree[lb.level][lb.box].list3Numbers[0] = 4*k;
					tree[lb.level][lb.box].list3Numbers[1] = 4*k+1;
					tree[lb.level][lb.box].list3Numbers[19] = 4*k+3;

					tree[j+1][4*k].list4Numbers[10] = lb.box;
					tree[j+1][4*k+1].list4Numbers[11] = lb.box;
					tree[j+1][4*k+3].list4Numbers[9] = lb.box;
				}
			}
			else {
				int pN = k/4;
				if (k == pN*4+2) {
					tree[lb.level][lb.box].list1_largeNumbers[0] = pN;
				}
				else if (k == pN*4+3) {
					tree[lb.level][lb.box].list1_largeNumbers[1] = pN;					
				}
				else if (k == pN*4+1) {
					tree[lb.level][lb.box].list1_largeNumbers[11] = pN;
				}					
			}
		}
		//}
		



		//	NEIGHBOR 1
		
		k = tree[lb.level][lb.box].neighborNumbers[1]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			if (tree[j][k].exists) {
				if (tree[j+1][4*k].exists) {	
					tree[lb.level][lb.box].list1_smallNumbers[1] = 4*k+3;	
					tree[lb.level][lb.box].list1_smallNumbers[2] = 4*k+2;
			
					tree[lb.level][lb.box].list3Numbers[2] = 4*k;
					tree[lb.level][lb.box].list3Numbers[3] = 4*k+1;	

					tree[j+1][4*k].list4Numbers[12] = lb.box;
					tree[j+1][4*k+1].list4Numbers[13] = lb.box;
					
				}
			}
			else {
				int pN = k/4;
				if (k == pN*4+2) {
					tree[lb.level][lb.box].list1_largeNumbers[1] = pN;
				}
				else if (k == pN*4+3) {
					tree[lb.level][lb.box].list1_largeNumbers[2] = pN;					
				}
			}
		}




		//	NEIGHBOR 2
		
		k = tree[lb.level][lb.box].neighborNumbers[2]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			if (tree[j][k].exists) {
				if (tree[j+1][4*k].exists) {	
					tree[lb.level][lb.box].list1_smallNumbers[3] = 4*k+3;	
		
					tree[lb.level][lb.box].list3Numbers[4] = 4*k;
					tree[lb.level][lb.box].list3Numbers[5] = 4*k+1;
					tree[lb.level][lb.box].list3Numbers[6] = 4*k+2;

					tree[j+1][4*k].list4Numbers[14] = lb.box;
					tree[j+1][4*k+1].list4Numbers[15] = lb.box;
					tree[j+1][4*k+2].list4Numbers[16] = lb.box;
				}
			}
			else {
				int pN = k/4;
				if (k == pN*4) {
					tree[lb.level][lb.box].list1_largeNumbers[4] = pN;
				}
				else if (k == pN*4+3) {
					tree[lb.level][lb.box].list1_largeNumbers[3] = pN;					
				}
			}
		}




		//	NEIGHBOR 3
		
		k = tree[lb.level][lb.box].neighborNumbers[3]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			if (tree[j][k].exists) {
				if (tree[j+1][4*k].exists) {	
					tree[lb.level][lb.box].list1_smallNumbers[4] = 4*k;
					tree[lb.level][lb.box].list1_smallNumbers[5] = 4*k+3;

					tree[lb.level][lb.box].list3Numbers[7] = 4*k+1;
					tree[lb.level][lb.box].list3Numbers[8] = 4*k+2;	

					tree[j+1][4*k+1].list4Numbers[17] = lb.box;
					tree[j+1][4*k+2].list4Numbers[18] = lb.box;
				}
			}
			else {
				int pN = k/4;
				if (k == pN*4) {
					tree[lb.level][lb.box].list1_largeNumbers[5] = pN;
				}
				else if (k == pN*4+3) {
					tree[lb.level][lb.box].list1_largeNumbers[4] = pN;					
				}
			}
		}




		//	NEIGHBOR 4
		
		k = tree[lb.level][lb.box].neighborNumbers[4]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			if (tree[j][k].exists) {
				if (tree[j+1][4*k].exists) {	
					tree[lb.level][lb.box].list1_smallNumbers[6] = 4*k;

					tree[lb.level][lb.box].list3Numbers[9] = 4*k+1;
					tree[lb.level][lb.box].list3Numbers[10] = 4*k+2;
					tree[lb.level][lb.box].list3Numbers[11] = 4*k+3;

					tree[j+1][4*k+1].list4Numbers[19] = lb.box;
					tree[j+1][4*k+2].list4Numbers[0] = lb.box;
					tree[j+1][4*k+3].list4Numbers[1] = lb.box;
				}
			}
			else {
				int pN = k/4;
				if (k == pN*4) {
					tree[lb.level][lb.box].list1_largeNumbers[6] = pN;
				}
				else if (k == pN*4+1) {
					tree[lb.level][lb.box].list1_largeNumbers[7] = pN;					
				}
			}
		}




		//	NEIGHBOR 5
		
		k = tree[lb.level][lb.box].neighborNumbers[5]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			if (tree[j][k].exists) {
				if (tree[j+1][4*k].exists) {	
					tree[lb.level][lb.box].list1_smallNumbers[7] = 4*k+1;
					tree[lb.level][lb.box].list1_smallNumbers[8] = 4*k;

					tree[lb.level][lb.box].list3Numbers[12] = 4*k+2;
					tree[lb.level][lb.box].list3Numbers[13] = 4*k+3;

					tree[j+1][4*k+2].list4Numbers[2] = lb.box;
					tree[j+1][4*k+3].list4Numbers[3] = lb.box;
				}
			}
			else {
				int pN = k/4;
				if (k == pN*4+1) {
					tree[lb.level][lb.box].list1_largeNumbers[8] = pN;
				}
				else if (k == pN*4) {
					tree[lb.level][lb.box].list1_largeNumbers[7] = pN;					
				}
			}
		}


	


		//	NEIGHBOR 6
		
		k = tree[lb.level][lb.box].neighborNumbers[6]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			if (tree[j][k].exists) {
				if (tree[j+1][4*k].exists) {	
					tree[lb.level][lb.box].list1_smallNumbers[9] = 4*k+1;

					tree[lb.level][lb.box].list3Numbers[16] = 4*k;
					tree[lb.level][lb.box].list3Numbers[14] = 4*k+2;
					tree[lb.level][lb.box].list3Numbers[15] = 4*k+3;

					tree[j+1][4*k].list4Numbers[6] = lb.box;
					tree[j+1][4*k+2].list4Numbers[4] = lb.box;
					tree[j+1][4*k+3].list4Numbers[5] = lb.box;
				}
			}
			else {
				int pN = k/4;
				if (k == pN*4+1) {
					tree[lb.level][lb.box].list1_largeNumbers[9] = pN;
				}
				else if (k == pN*4+2) {
					tree[lb.level][lb.box].list1_largeNumbers[10] = pN;					
				}
			}
		}


		//	NEIGHBOR 7
		
		k = tree[lb.level][lb.box].neighborNumbers[7]; // neighbor is in same level
		j = lb.level;
		if (k != -1) { // k=-1 means that box is not in the computational domain
			if (tree[j][k].exists) {
				if (tree[j+1][4*k].exists) {	
					tree[lb.level][lb.box].list1_smallNumbers[10] = 4*k+2;
					tree[lb.level][lb.box].list1_smallNumbers[11] = 4*k+1;

					tree[lb.level][lb.box].list3Numbers[18] = 4*k;
					tree[lb.level][lb.box].list3Numbers[17] = 4*k+3;

					tree[j+1][4*k].list4Numbers[8] = lb.box;
					tree[j+1][4*k+3].list4Numbers[7] = lb.box;
				}
			}
			else {
			}
		}

	}


	

	//	Obtain the desired matrix
	void obtain_Desired_Operator(std::vector<pts2D>& shiftedChebNodes, Eigen::MatrixXd& T) {
		T	=	Eigen::MatrixXd(rank,rank);
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				T(i,j)	=	K->getInteraction(standardChebNodes[i], shiftedChebNodes[j], a);//+ boxLogHomogRadius[nLevels];
			}
		}
	}

	void obtain_Self_Operator(Eigen::MatrixXd& T) {
		T	=	Eigen::MatrixXd(rank,rank);
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				T(i,j)	=	K->getInteraction(standardChebNodes[i], standardChebNodes[j], a);
			}
		}
	}

	//	Assemble FMM Operators
	void assemble_Operators_FMM() {
		std::vector<pts2D> shiftedChebNodes;
		//	Assemble Outer Interactions
		for (int l=0; l<6; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(l-3,-3);
			obtain_Desired_Operator(shiftedChebNodes, M2LOuter[l]);
		}
		for (int l=0; l<6; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(3,l-3);
			obtain_Desired_Operator(shiftedChebNodes, M2LOuter[l+6]);
		}
		for (int l=0; l<6; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(3-l,3);
			obtain_Desired_Operator(shiftedChebNodes, M2LOuter[l+12]);
		}
		for (int l=0; l<6; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(-3,3-l);
			obtain_Desired_Operator(shiftedChebNodes, M2LOuter[l+18]);
		}
		//	Assemble Inner Interactions
		for (int l=0; l<4; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(l-2,-2);
			obtain_Desired_Operator(shiftedChebNodes, M2LInner[l]);
		}
		for (int l=0; l<4; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(2,l-2);
			obtain_Desired_Operator(shiftedChebNodes, M2LInner[l+4]);
		}
		for (int l=0; l<4; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(2-l,2);
			obtain_Desired_Operator(shiftedChebNodes, M2LInner[l+8]);
		}
		for (int l=0; l<4; ++l) {
			shiftedChebNodes	=	shift_Cheb_Nodes(-2,2-l);
			obtain_Desired_Operator(shiftedChebNodes, M2LInner[l+12]);
		}


		//	Assemble Neighbor Interactions
		{
			shiftedChebNodes	=	shift_Cheb_Nodes(-1.0,-1.0);
			obtain_Desired_Operator(shiftedChebNodes, neighborInteraction[0]);
		}
		{
			shiftedChebNodes	=	shift_Cheb_Nodes(0.0,-1.0);
			obtain_Desired_Operator(shiftedChebNodes, neighborInteraction[1]);
		}
		{
			shiftedChebNodes	=	shift_Cheb_Nodes(1.0,-1.0);
			obtain_Desired_Operator(shiftedChebNodes, neighborInteraction[2]);
		}
		{
			shiftedChebNodes	=	shift_Cheb_Nodes(1.0,0.0);
			obtain_Desired_Operator(shiftedChebNodes, neighborInteraction[3]);
		}
		{
			shiftedChebNodes	=	shift_Cheb_Nodes(1.0,1.0);
			obtain_Desired_Operator(shiftedChebNodes, neighborInteraction[4]);
		}
		{
			shiftedChebNodes	=	shift_Cheb_Nodes(0.0,1.0);
			obtain_Desired_Operator(shiftedChebNodes, neighborInteraction[5]);
		}
		{
			shiftedChebNodes	=	shift_Cheb_Nodes(-1.0,1.0);
			obtain_Desired_Operator(shiftedChebNodes, neighborInteraction[6]);
		}
		{
			shiftedChebNodes	=	shift_Cheb_Nodes(-1.0,0.0);
			obtain_Desired_Operator(shiftedChebNodes, neighborInteraction[7]);
		}
		//	Assemble Self Interactions
		{
			obtain_Self_Operator(selfInteraction);
		}

		//List1_Small_Interaction
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.5,-1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[0]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-0.5,-1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[1]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(0.5,-1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[2]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.5,-1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[3]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.5,-0.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[4]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.5,0.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[5]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.5,1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[6]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(0.5,1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[7]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-0.5,1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[8]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.5,1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[9]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.5,0.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[10]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.5,-0.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List1_Small_Interaction[11]);
		}

		//	List1_Large_Interaction
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-3.0,-3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[0]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.0,-3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[1]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.0,-3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[2]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(3.0,-3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[3]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(3.0,-1.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[4]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(3.0,1.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[5]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(3.0,3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[6]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.0,3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[7]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.0,3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[8]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-3.0,3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[9]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-3.0,1.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[10]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-3.0,-1.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List1_Large_Interaction[11]);
		}
		

		//	List4_Interaction
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-5.0,-5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[0]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-3.0,-5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[1]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.0,-5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[2]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.0,-5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[3]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(3.0,-5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[4]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(5.0,-5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[5]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(5.0,-3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[6]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(5.0,-1.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[7]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(5.0,1.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[8]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(5.0,3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[9]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(5.0,5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[10]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(3.0,5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[11]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.0,5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[12]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.0,5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[13]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-3.0,5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[14]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-5.0,5.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[15]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-5.0,3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[16]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-5.0,1.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[17]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-5.0,-1.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[18]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-5.0,-3.0,2);
			obtain_Desired_Operator(shiftedChebNodes, List4_Interaction[19]);
		}
		

		//	List3_Interaction
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-2.5,-2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[0]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.5,-2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[1]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-0.5,-2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[2]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(0.5,-2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[3]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.5,-2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[4]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(2.5,-2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[5]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(2.5,-1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[6]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(2.5,-0.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[7]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(2.5,0.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[8]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(2.5,1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[9]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(2.5,2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[10]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(1.5,2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[11]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(0.5,2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[12]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-0.5,2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[13]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-1.5,2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[14]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-2.5,2.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[15]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-2.5,1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[16]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-2.5,0.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[17]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-2.5,-0.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[18]);
		}
		{
			shiftedChebNodes	=	shift_scale_Cheb_Nodes(-2.5,-1.5,0.5);
			obtain_Desired_Operator(shiftedChebNodes, List3_Interaction[19]);
		}
	}



	//	assign_Leaf_ChebNodes
	void assign_Leaf_ChebNodes() {
		for (int j=0; j<childless_boxes.size(); ++j) {
			level_box lb = childless_boxes[j];
			tree[lb.level][lb.box].chebNodes	=	shift_scale_Cheb_Nodes(tree[lb.level][lb.box].center.x,tree[lb.level][lb.box].center.y,boxRadius[lb.level]);
		}
	}




	// evaluate multipoles
	void evaluate_multipoles() {
		for (int j=0; j<childless_boxes.size(); ++j) { // leafs
			level_box lb = childless_boxes[j];
			tree[lb.level][lb.box].multipoles = Eigen::VectorXd::Zero(rank);
			for (int m=0; m<rank; ++m) { // multipoles
				for (int i=0; i<tree[lb.level][lb.box].charge_indices.size(); ++i) { // charges in leaf
					tree[lb.level][lb.box].multipoles[m] = tree[lb.level][lb.box].multipoles[m] + charge_database[tree[lb.level][lb.box].charge_indices[i]].q * get_S(tree[lb.level][lb.box].chebNodes[m].x, charge_database[tree[lb.level][lb.box].charge_indices[i]].x, nChebNodes) * get_S(tree[lb.level][lb.box].chebNodes[m].y, charge_database[tree[lb.level][lb.box].charge_indices[i]].y, nChebNodes);
				}
			}
		}
	}



	void evaluate_All_M2M() {
		for (int j=nLevels-1; j>0; --j) { // parent
			int J	=	j+1; // children
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) { // parent
				int K	=	4*k; // children
				if (tree[J][K].exists) {
					tree[j][k].multipoles	=	M2M[0]*tree[J][K].multipoles+M2M[1]*tree[J][K+1].multipoles+M2M[2]*tree[J][K+2].multipoles+M2M[3]*tree[J][K+3].multipoles;
				}
			}
		}
	}

	void evaluate_list2() { // list 2
		for (int j=0; j<=1; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				if (tree[j][k].exists) {
					tree[j][k].locals	=	Eigen::VectorXd::Zero(rank);
				}
			}
		}
		for (int j=2; j<=nLevels; ++j) {
			//#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				if (tree[j][k].exists) {
					//cout << endl << "j: " << j << "	k: " << k << endl;
					tree[j][k].locals	=	Eigen::VectorXd::Zero(rank);
					#ifdef HOMOG
						//	Inner well-separated clusters
						for (int l=0; l<16; ++l) {
							int nInner	=	tree[j][k].innerNumbers[l];
							if (nInner>-1) {
								tree[j][k].locals+=M2LInner[l]*tree[j][nInner].multipoles;
							}
						}
						//	Outer well-separated clusters
						for (int l=0; l<24; ++l) {
							int nOuter	=	tree[j][k].outerNumbers[l];
							if (nOuter>-1) {
								tree[j][k].locals+=M2LOuter[l]*tree[j][nOuter].multipoles;
							}
						}
						tree[j][k].locals*=boxHomogRadius[j];					
					#elif LOGHOMOG
						//	Inner well-separated clusters
						for (int l=0; l<16; ++l) {
							int nInner	=	tree[j][k].innerNumbers[l];
							if (nInner>-1) {
								//cout << "l: " << l << "	nInner: " << nInner << endl;
								tree[j][k].locals+=M2LInner[l]*tree[j][nInner].multipoles;
								tree[j][k].locals+=boxLogHomogRadius[j]*tree[j][nInner].multipoles.sum()*Eigen::VectorXd::Ones(rank);
							}
						}
						//	Outer well-separated clusters
						for (int l=0; l<24; ++l) {
							int nOuter	=	tree[j][k].outerNumbers[l];
							if (nOuter>-1) {
								//cout << "l: " << l << "	nOuter: " << nOuter << endl;
								tree[j][k].locals+=M2LOuter[l]*tree[j][nOuter].multipoles;
								tree[j][k].locals+=boxLogHomogRadius[j]*tree[j][nOuter].multipoles.sum()*Eigen::VectorXd::Ones(rank);
							}
						}
					#endif
				}
			}
		}
	}




	void evaluate_list3() {
		for (int i=0; i<childless_boxes.size(); ++i) { // all childless boxes
			int j = childless_boxes[i].level; // box b
			int k = childless_boxes[i].box;
			#ifdef HOMOG
			for (int l=0; l<20; ++l) {
				int nList3	=	tree[j][k].list3Numbers[l];
				if (nList3>-1) {
					tree[j][k].locals+=boxHomogRadius[j]*List3_Interaction[l]*tree[j+1][nList3].multipoles;
				}
			}
			#elif LOGHOMOG
			
			for (int l=0; l<20; ++l) {
				int nList3	=	tree[j][k].list3Numbers[l];
				if (nList3>-1) {
					//cout << "l: " << l << "	nList3: " << nList3 << endl;
					tree[j][k].locals+=List3_Interaction[l]*tree[j+1][nList3].multipoles;
					tree[j][k].locals+=boxLogHomogRadius[j]*tree[j+1][nList3].multipoles.sum()*Eigen::VectorXd::Ones(rank);
				}
			}
			#endif
		}
	}



	void evaluate_list4() {
		for (int j=2; j<=nLevels; ++j) { // all boxes // box b
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				if (tree[j][k].exists) {
					#ifdef HOMOG
					for (int l=0; l<20; ++l) {
						int nList4	=	tree[j][k].list4Numbers[l];
						if (nList4>-1) {
							tree[j][k].locals+=boxHomogRadius[j]*List4_Interaction[l]*tree[j-1][nList4].multipoles;
						}
					}
					#elif LOGHOMOG
					
					for (int l=0; l<20; ++l) {
						int nList4	=	tree[j][k].list4Numbers[l];
						if (nList4>-1) {
							tree[j][k].locals+=List4_Interaction[l]*tree[j-1][nList4].multipoles;
							tree[j][k].locals+=boxLogHomogRadius[j]*tree[j-1][nList4].multipoles.sum()*Eigen::VectorXd::Ones(rank);
						}
					}
					#endif
				}
			}
		}
	}




	void evaluate_All_L2L() {
		for (int j=2; j<nLevels; ++j) {
			int J	=	j+1;
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				int K	=	4*k;
				if (tree[J][K].exists) { // if children exist
					tree[J][K].locals+=L2L[0]*tree[j][k].locals;
					tree[J][K+1].locals+=L2L[1]*tree[j][k].locals;
					tree[J][K+2].locals+=L2L[2]*tree[j][k].locals;
					tree[J][K+3].locals+=L2L[3]*tree[j][k].locals;
				}
			}
		}
	}





//	Self & Neighbor Interaction
	void evaluate_list1() {
		for (int i=0; i<childless_boxes.size(); ++i) { // all childless boxes
			int j = childless_boxes[i].level; // box b
			int k = childless_boxes[i].box;
			#ifdef HOMOG
			//	List1_Small_Interaction
			for (int l=0; l<12; ++l) {
				int nList1_small	=	tree[j][k].list1_smallNumbers[l];
				if (nList1_small>-1) {
					tree[j][k].locals+=boxHomogRadius[j]*List1_Small_Interaction[l]*tree[j+1][nList1_small].multipoles;
				}
			}
			//	List1_Large_Interaction
			for (int l=0; l<12; ++l) {
				int nList1_large	=	tree[j][k].list1_largeNumbers[l];
				if (nList1_large>-1) {
					tree[j][k].locals+=boxHomogRadius[j]*List1_Large_Interaction[l]*tree[j-1][nList1_large].multipoles;
				}
			}
			//	neighborInteraction
			for (int l=0; l<8; ++l) {
				int nNeighbor	=	tree[j][k].neighborNumbers[l];
				if (nNeighbor>-1) {
					if (tree[j][nNeighbor].exists && !tree[j+1][4*nNeighbor].exists) {
						tree[j][k].locals+=boxHomogRadius[j]*neighborInteraction[l]*tree[j][nNeighbor].multipoles;
					}
				}
			}
			//	Self Interaction
			tree[j][k].locals+=boxHomogRadius[j]*selfInteraction*tree[j][k].multipoles;

			#elif LOGHOMOG
			//	List1_Small_Interaction
			for (int l=0; l<12; ++l) {
				int nList1_small	=	tree[j][k].list1_smallNumbers[l];
				if (nList1_small>-1) {
					tree[j][k].locals+=List1_Small_Interaction[l]*tree[j+1][nList1_small].multipoles;
					tree[j][k].locals+=boxLogHomogRadius[j]*tree[j+1][nList1_small].multipoles.sum()*Eigen::VectorXd::Ones(rank);
				}
			}
			//	List1_Large_Interaction
			for (int l=0; l<12; ++l) {
				int nList1_large	=	tree[j][k].list1_largeNumbers[l];
				if (nList1_large>-1) {
					tree[j][k].locals+=List1_Large_Interaction[l]*tree[j-1][nList1_large].multipoles;
					tree[j][k].locals+=boxLogHomogRadius[j]*tree[j-1][nList1_large].multipoles.sum()*Eigen::VectorXd::Ones(rank);
					//cout << "L1: " << nList1_large << endl;
				}
			}
			//	neighborInteraction
			for (int l=0; l<8; ++l) {
				int nNeighbor	=	tree[j][k].neighborNumbers[l];
				if (nNeighbor>-1) {
					if (tree[j][nNeighbor].exists && !tree[j+1][4*nNeighbor].exists) {
						tree[j][k].locals+=neighborInteraction[l]*tree[j][nNeighbor].multipoles;
						tree[j][k].locals+=boxLogHomogRadius[j]*tree[j][nNeighbor].multipoles.sum()*Eigen::VectorXd::Ones(rank);
					}
				}
			}
			//	Self Interaction
			Eigen::MatrixXd current_selfInteraction = Eigen::MatrixXd(rank,rank);
			for (int c=0; c<rank; ++c) {
				for (int d=0; d<rank; ++d) {
					if (c!=d) {
						current_selfInteraction(c,d) = selfInteraction(c,d) + boxLogHomogRadius[j];
					}
					else {
						current_selfInteraction(c,d) = selfInteraction(c,d);
					}						
				}
			}
			tree[j][k].locals += current_selfInteraction*tree[j][k].multipoles;
			#endif
		}
	}




	void perform_Error_Check() {
		srand (time(NULL));
		int c 	=	rand()%childless_boxes.size();

		level_box nBox = childless_boxes[c];
		
		Eigen::VectorXd potential	=	Eigen::VectorXd::Zero(rank);
		for (int l1=0; l1<rank; ++l1) {// cheb nodes of nBox
			for (int k=0; k<childless_boxes.size(); ++k) { // other boxes which includes nBox
				level_box other_boxes = childless_boxes[k];

				/*for (int l2=0; l2<tree[other_boxes.level][other_boxes.box].charge_indices.size(); ++l2) { // cheb nodes of other boxes or charge locations of charges in other boxes
					pts2D charge_loc;
					charge_loc.x = charge_database[tree[other_boxes.level][other_boxes.box].charge_indices[l2]].x;
					charge_loc.y = charge_database[tree[other_boxes.level][other_boxes.box].charge_indices[l2]].y;
					potential(l1)+=K->getInteraction(tree[nBox.level][nBox.box].chebNodes[l1], charge_loc, a)*charge_database[tree[other_boxes.level][other_boxes.box].charge_indices[l2]].q;
				}*/

				for (int l2=0; l2<rank; ++l2) { // cheb nodes of other boxes or charge locations of charges in other boxes (using multipoles)
					potential(l1)+=K->getInteraction(tree[nBox.level][nBox.box].chebNodes[l1], tree[other_boxes.level][other_boxes.box].chebNodes[l2], a)*tree[other_boxes.level][other_boxes.box].multipoles(l2);
				}
			}
		}
		Eigen::VectorXd error(rank);
		for (int k=0; k<rank; ++k) {
			error(k)	=	fabs((potential-tree[nBox.level][nBox.box].locals)(k)/potential(k));
		}
		cout << "nBox.level: " << nBox.level << " nBox.box: " << nBox.box << " er: "<< error.maxCoeff() << endl;
		
		
		//return error.maxCoeff();
	}


	void check4() {
		for (int i=0; i<childless_boxes.size(); ++i) { // all childless boxes
			int j = childless_boxes[i].level; // box b
			int k = childless_boxes[i].box;


	/*for (int j=2; j<=nLevels; ++j) { // all boxes // box b
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				if (tree[j][k].exists) {*/
			cout << endl << "j: " << j << "	k: " << k << endl;
			
			cout << "neighborNumbers" << endl;
			for (int l=0; l<8; ++l) {
				int nN = tree[j][k].neighborNumbers[l];
				if(nN!=-1 && tree[j][nN].exists && !tree[j+1][4*nN].exists) {
					cout << "l: " << l << "	" << tree[j][k].neighborNumbers[l] << endl;
				}
			}
			/*cout << endl << "List4" << endl;
			for (int l=0; l<16; ++l) {
				cout << tree[j][k].List4Numbers[l]<<",";
			}*/
			
			cout << endl;
		}
	}
//}
//}

};

#endif
