#include<stdio.h>
#include<stdlib.h>
#include "mrc.h"
#include"cufunc.h"
#include<string.h>

//#include<shift.cu>


void printmrc(float *mrcdata,int nx)
{
        for(int i=0;i<10;i++)
        {
                for(int j=0;j<10;j++)
                        printf("%.2f ",mrcdata[i*nx+j]);
                printf("\n");
        }
}
cufftComplex muiltdata(cufftComplex a,cufftComplex b)
{
        cufftComplex res;
        res.x=a.x*b.x-a.y*b.y;
        res.y=a.x*b.y+a.y*b.x;
        return res;
}
//float bfactor=-2;
void MkPosList(MASK *list, DIM nsam, float bfactor)
{
        int hnsamxb=nsam.x/2+1;//2461
        int hnsamy=nsam.y/2;//1830
        int i,j;
        int count=0;
        float r2;
        float m=-0.5*bfactor;//-0.5*150=-75
        DIM nsam2=nsam*nsam;//4920*4920 ; 3660*3660

        for(j=0;j<hnsamy;j++)//1830
                for(i=0;i<hnsamxb;i++)//2461
                {
                        list[count].x=i;
                        list[count].y=j;

                        r2=list[count].x*list[count].x/float(nsam2.x)+list[count].y*list[count].y/float(nsam2.y);
                        if((list[count].x+list[count].y)%2==0) list[count].z=exp(m*r2);
                        else list[count].z=-exp(m*r2);

                        count++;
                }//end count=4503630
        for(j=hnsamy;j<nsam.y;j++)//[1830 : 3660]
                for(i=0;i<hnsamxb;i++)//[0 : 2461]
                {
                        list[count].x=i;
                        list[count].y=j-nsam.height();
                        r2=list[count].x*list[count].x/float(nsam2.x)+list[count].y*list[count].y/float(nsam2.y);
                        if((list[count].x+list[count].y)%2==0) list[count].z=exp(m*r2);
                        else list[count].z=-exp(m*r2);

                        count++;
                }

}


int main()
{

        MRC stack;
        MRC stack2;
        //stack.open("Rawimage.mrc","rb"); //nx=228;ny=269
        //stack.open("Rawimage.mrc","rb"); //nx=228;ny=269

        stack.open("shift1.mrc","rb"); //nx=228;ny=269
        stack2.open("shift2.mrc","rb"); //nx=228;ny=269

        int nx=stack.getNx();
        int ny=stack.getNy();


        int sizenum=nx*ny;

        DIM nsamUnbin(nx,ny);


        float *bufmrc=new float[sizenum];
        if(stack.read2DIm_32bit(bufmrc,0)!=stack.getImSize())
                printf("Some error");

/*      MRC stack3;
        stack3.open("data3.mrc","wb");
        stack3.createMRC(bufmrc,nsamUnbin.width(),nsamUnbin.height(),1);
        stack3.close();*/

        cufftComplex *tempComplex;
        cufftComplex *cpudata=new cufftComplex[sizenum];
        cufftComplex *tempComplex2=new cufftComplex[sizenum];

        int flag=0;
        for(int i=0;i<ny;i++)
                for(int j=0;j<nx;j++)
                {
                        cpudata[i*nx+j].x=bufmrc[i*nx+j];
                        cpudata[i*nx+j].y=0;
                        if(bufmrc[i*nx+j]==1&&flag==0)
                        {
                                printf(" cor %d %d \n",j,i);
                                flag=1;
                        }
                }


        cufftComplex *dfft=0;
        GPUMemAlloc((void **)&dfft,sizeof(cufftComplex)*sizenum);
        GPUMemAlloc((void **)&tempComplex,sizeof(cufftComplex)*sizenum);

        GPUMemZero((void **)&dfft,sizeof(cufftComplex)*sizenum);


        //make a list
//      int sizec=(nx/2+1)*ny;//(2461+1)*3660
/*      MASK *hPosList=new MASK[sizec];
        MASK *dPosList=0;
        MkPosList(hPosList,nsam,10);
        GPUMemAlloc((void **)&dPosList,sizeof(MASK)*sizec);
        GPUMemH2D((void **)dPosList,(void **)hPosList,sizeof(MASK)*sizec);*/



        ///
/*      for(int i=0;i<ny;i++)
        {
                memcpy(bufmrcfft+i*sizebx, bufmrc+i*nx, sizeof(float)*nx);
        }*/

        GPUMemH2D((cufftComplex *)dfft,(cufftComplex *)cpudata,sizeof(cufftComplex)*sizenum);
        GPUSync();

        //do fft
        cufftHandle fft_plan,ifft_plan;
        fft_plan=GPUFFTPlan(nsamUnbin);
        ifft_plan=GPUIFFTPlan(nsamUnbin);

        GPUSync();
        //GPUFFT2d(dfft,fft_plan);//every image fft
        GPUFFT2d2(dfft,tempComplex,fft_plan);
        GPUSync();
        GPUMemD2H((void *)cpudata,(void *)tempComplex,sizeof(cufftComplex)*sizenum);

/////////////////////////////shift
        GPUMemD2H((void *)tempComplex2,(void *)tempComplex,sizeof(cufftComplex)*sizenum);
                float shiftx=20;
                float shifty=20;
                        for(int i=0;i<ny;i++)
                                for(int j=0;j<nx;j++)
                                {
                                        int index=i*nx+j;
                                        //float t=(i*shiftx+j*shifty)/sizenum;
                                        float shx=shiftx*2*3.1415926/nx;
                                        float shy=shifty*2*3.1415926/ny;

                                        float t= j*shx+i*shy;
                                        //float t=i*shiftx/nx+j*shifty/ny;
                                        cufftComplex datatemp;
                                        datatemp.x=cos(t);
                                        datatemp.y=sin(t);
                                        tempComplex2[index]=muiltdata(tempComplex2[index],datatemp);
                                }

    GPUMemH2D((void *)tempComplex,(void *)tempComplex2,sizeof(cufftComplex)*sizenum);
    GPUIFFT2d(tempComplex,dfft,ifft_plan);
    GPUSync();
    GPUMultiplyNum(dfft,1.0/sizenum,sizenum);//inverse FFT
    GPUSync();
    GPUMemD2H((void *)cpudata,(void *)dfft,sizeof(cufftComplex)*sizenum);


    flag=0;
    float *bufmrcfft=new float[sizenum];
        for(int i=0;i<ny;i++)
                for(int j=0;j<nx;j++)
                {
                        bufmrcfft[i*nx+j]=cpudata[i*nx+j].x;
                        if(bufmrcfft[i*nx+j]+0.0001>1&&flag==0)
                                {
                                        printf(" cor %d %d \n",j,i);
                                        flag=1;
                                }
                }
        MRC stack2;
        stack2.open("FFTimage.mrc","wb");
        stack2.createMRC(bufmrcfft,nsamUnbin.width(),nsamUnbin.height(),1);
        stack2.close();
        //GPUShift(dfft,dPosList,-20,-20, nsam);
/*      cufftComplex *tempComplex2=new cufftComplex[sizeb];

        //GPUIFFT2d(dfft,ifft_plan);
        //GPUMultiplyNum(dfft,1.0/sizenum,sizenum);//inverse FFT
        //GPUSync();
        //GPUMemD2H((void *)bufmrcfft,(void *)dfft,sizeof(float)*sizeb);
        //GPUSync();
        //printmrc(bufmrcfft,nx+2);
/*      MRC stack2;
        stack2.open("FFTimage.mrc","wb");
        stack2.createMRC(bufmrcfft,nsamUnbin.width(),nsamUnbin.height(),1);
        stack2.close();*/






}
/*      cufftComplex *tempComplex2=new cufftComplex[sizenum];
        GPUMemD2H((void *)tempComplex2,(void *)tempComplex,sizeof(cufftComplex)*sizenum);
        float shiftx=30;
        float shifty=30;
                for(int i=0;i<nx;i++)
                        for(int j=0;j<ny;j++)
                        {
                                int index=i*ny+j;
                                //float t=(i*shiftx+j*shifty)/sizenum;
                                float shx=shiftx*2*3.1415926/nx;
                                float shy=shifty*2*3.1415926/ny;

                                float t= i*shx+j*shy;
                                //float t=i*shiftx/nx+j*shifty/ny;
                                cufftComplex datatemp;
                                datatemp.x=cos(t);
                                datatemp.y=sin(t);
                                tempComplex2[index]=muiltdata(tempComplex2[index],datatemp);
                        }

        GPUMemH2D((void *)tempComplex,(void *)tempComplex2,sizeof(cufftComplex)*sizenum);
*/
