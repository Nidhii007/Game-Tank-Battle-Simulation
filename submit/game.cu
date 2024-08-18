    #include <iostream>
    #include <stdio.h>
    #include <cuda.h>
    #include <chrono>
    #include <algorithm>
    #include<thrust/device_vector.h>
    #include<thrust/extrema.h>
    #include<thrust/count.h>
    using namespace std;

    //*******************************************

    // Write down the kernels here
    __device__ 
    long long gcd(long long int a, long long int b){
        if(b==0)    return a;
        return gcd(b,a%b);
    }

    __global__ void findFire(int k,int *xcoord, int *ycoord, int *HP, int T, int *min_m, int *index){
        if(k%T!=0){
            int id=blockIdx.x;
            int ind=threadIdx.x;
            __shared__ int mdx;
            __shared__ int mdy;
            __shared__ int minm[1024];
            __shared__ int min_index[1024];
            minm[ind]=INT_MAX;
            min_index[ind]=ind;
            if(ind==0){
                int j=(id+k)%T;
                mdx=xcoord[j]-xcoord[id];
                mdy=ycoord[j]-ycoord[id];
                int gcd1=abs(gcd(mdx,mdy));
                mdx/=gcd1;
                mdy/=gcd1;
                minm[T]=INT_MAX;
                index[id]=-1;
            }
            __syncthreads();

            //For finding all tanks in the direction of fire from the tank id 
            if(ind<T && HP[id]>0 && ind!=id && HP[ind]>0){
                int dx=xcoord[ind]-xcoord[id];
                int dy=ycoord[ind]-ycoord[id];
                int dydir=dy;
                int dxdir=dx;
                int gcd2=abs(gcd(dx,dy));
                dxdir=dx/gcd2;
                dydir=dy/gcd2;
                if(mdx==dxdir && mdy==dydir){
                    minm[ind]=(mdy==0?(dx/mdx):(dy/mdy));
                }
            }
            __syncthreads();
    
            if(HP[id]>0){
                //For taking minimum distance(in the direction of fire) from the tank id 
                // using reduction 
                for(int s=blockDim.x/2;s>0;s>>=1){
                    if(ind<s){
                        if(minm[ind]>minm[ind+s]){
                            minm[ind]=minm[ind+s];
                            min_index[ind]=min_index[ind+s];
                        }   
                    }
                    __syncthreads();
                }
                __syncthreads();
                if(ind==0){
                    if(minm[0]!=INT_MAX){
                    min_m[id]=minm[0];
                    index[id]=min_index[0];
                    }
                }
            
            }
        }
        __syncthreads();
    }

    __global__ void updateScoreHP(int k, int T, int *index, int *HP, int *score, int *activeTanks){
            int id=threadIdx.x;
            if(k%T!=0 && index[id]!=-1){
                atomicAdd(&HP[index[id]],-1);
                atomicAdd(&score[id],1);
            }
            __syncthreads();
            if(id==0)
                *activeTanks=thrust::count_if(thrust::cuda::par, HP, HP+T,[] __device__ (int hp){return hp>0;});
    }

    //***********************************************


    int main(int argc,char **argv)
    {
        // Variable declarations
        int M,N,T,H,*xcoord,*ycoord,*score;
        

        FILE *inputfilepointer;
        
        //File Opening for read
        char *inputfilename = argv[1];
        inputfilepointer    = fopen( inputfilename , "r");

        if ( inputfilepointer == NULL )  {
            printf( "input.txt file failed to open." );
            return 0; 
        }

        fscanf( inputfilepointer, "%d", &M );
        fscanf( inputfilepointer, "%d", &N );
        fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
        fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
        
        // Allocate memory on CPU
        xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
        ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
        score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

        // Get the Input of Tank coordinates
        for(int i=0;i<T;i++)
        {
        fscanf( inputfilepointer, "%d", &xcoord[i] );
        fscanf( inputfilepointer, "%d", &ycoord[i] );
        }
            

        auto start = chrono::high_resolution_clock::now();

        //*********************************
        // Your Code begins here (Do not change anything in main() above this comment)
        //********************************

        int *dxcoord;
        int *dycoord;
        cudaMalloc(&dxcoord , sizeof(int) * T) ;
        cudaMemcpy(dxcoord,xcoord,sizeof(int) * T,cudaMemcpyHostToDevice);
        cudaMalloc(&dycoord , sizeof(int) * T) ;
        cudaMemcpy(dycoord,ycoord,sizeof(int) * T,cudaMemcpyHostToDevice);
        
        int *dscore;
        memset(score, 0, sizeof(int) * T);
        cudaMalloc(&dscore , sizeof(int) * T) ;
        cudaMemcpy(dscore,score,sizeof(int) * T,cudaMemcpyHostToDevice);

        thrust::device_vector<int> dHP(T,H);
        int *HPptr = thrust::raw_pointer_cast(dHP.data());

        int *min_m;
        int *dindex;
        int *activeTanks;

        cudaMalloc(&min_m, sizeof(int)*T);
        cudaMalloc(&dindex, sizeof(int)*T);
        cudaHostAlloc(&activeTanks, sizeof(int),0);
        *activeTanks=T;
       
        int blockSize=pow(2,ceil(log(T)/log(2)));

        for(int k=1;*activeTanks>1;k++){
            findFire<<<T,blockSize>>>(k, dxcoord, dycoord, HPptr, T, min_m, dindex);
            cudaDeviceSynchronize();
            updateScoreHP<<<1,T>>>(k, T, dindex, HPptr, dscore, activeTanks);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(score,dscore,sizeof(int) * T,cudaMemcpyDeviceToHost);


        //*********************************
        // Your Code ends here (Do not change anything in main() below this comment)
        //********************************

        auto end  = chrono::high_resolution_clock::now();

        chrono::duration<double, std::micro> timeTaken = end-start;

        printf("Execution time : %f\n", timeTaken.count());

        // Output
        char *outputfilename = argv[2];
        char *exectimefilename = argv[3]; 
        FILE *outputfilepointer;
        outputfilepointer = fopen(outputfilename,"w");

        for(int i=0;i<T;i++)
        {
            fprintf( outputfilepointer, "%d\n", score[i]);
        }
        fclose(inputfilepointer);
        fclose(outputfilepointer);

        outputfilepointer = fopen(exectimefilename,"w");
        fprintf(outputfilepointer,"%f", timeTaken.count());
        fclose(outputfilepointer);

        free(xcoord);
        free(ycoord);
        free(score);
        cudaDeviceSynchronize();
        return 0;
    }   
