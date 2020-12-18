#include <iostream>
#include <iomanip>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <cmath>

double acceleration(double t)
{
  return sin(t);
}

void calc(double* trace, uint32_t traceSize, double t0, double dt, double y0, double y1, int rank, int size)
{
  MPI_Status status;
  MPI_Bcast(&traceSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&t0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  uint32_t first = traceSize * rank/size;
  uint32_t last = traceSize * (rank + 1)/size;
  uint32_t l = last - first;
  double v0 = 0.0;
  double sy = 0.0, ey = 0.0, py = 0.0, sv = 0.0, ev = 0.0, pv = 0.0;
  if(!rank){
    sy = y0;
  }
  double* loc_trace = (double*)calloc(len, sizeof(*loc_trace));
  if(!loc_trace){
    exit(0);
  }
  double tt = dt *traceSize/size;
  t0 = t0 + tt * rank;

  // Sighting shot
  loc_trace[0] = sy;
  loc_trace[1] = sy + dt * sv;
  for(uint32_t i = 2; i < l; i++){
    loc_trace[i] = dt * dt * acceleration (t0 + (i - 1) * dt) + 2 * loc_trace[i - 1] - loc_trace[i - 2];
  }
  ey = loc_trace[l - 1];
  ev = (loc_trace[l - 1] - loc_trace[l - 2])/dt;

  if(size == 1){
    v0 = (y1 - ey)/()dt * traceSize;
    sv = v0;
  }
  else{
    if(rank){
      MPI_Recv(&py, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&pv, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
      sy = py;
      sv = pv;
      ey += py + pv * tt;
      ev += pv;
    }
    if(rank != size - 1){
      MPI_Send(&ey, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
      MPI_Send(&ev, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
    }
    if(rank == size - 1){
      MPI_Send(&ey, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    if(!rank){
      MPI_Recv(&py, 1, MPI_DOUBLE, size - 1, 0, MPI_COMM_WORLD, &status);
      v0 = (y1 - py)/(dt * traceSize);
    }
    MPI_Bcast(&v0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    sy += v0 * tt * rank;
    sv += v0;
  }

  // The final shot
  loc_trace[0] = sy;
  loc_trace[1] = sy + dt * sv;
  for(uint32_t i = 2; i < l; i++){
    loc_trace[i] = dt * dt * acceleration (t0 + (i - 1) * dt) + 2 * loc_trace[i - 1] - loc_trace[i - 2];
  }
  if(!rank){
    memcpy(trace, loc_trace, l * sizeof(double));
    for(int i = 1; i < size; i++){
      uint32_t first = traceSize * i/size;
      uint32_t last = traceSize * (i + 1)/size;
      uint32_t len = last - first;
      MPI_Recv(trace + first, len, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
    }
  }
  if(rank){
    MPI_Send(loc_trace, l, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }
  free(loc_trace);

}

int main(int argc, char** argv)
{
  int rank = 0, size = 0, status = 0;
  uint32_t traceSize = 0;
  double t0 = 0, t1 = 0, dt = 0, y0 = 0, y1 = 0;
  double* trace = 0;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0)
  {
    // Check arguments
    if (argc != 3)
    {
      std::cout << "[Error] Usage <inputfile> <output file>\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Prepare input file
    std::ifstream input(argv[1]);
    if (!input.is_open())
    {
      std::cout << "[Error] Can't open " << argv[1] << " for write\n";
      status = 1;
      MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
      return 1;
    }

    // Read arguments from input
    input >> t0 >> t1 >> dt >> y0 >> y1;
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    traceSize = (t1 - t0)/dt;
    trace = new double[traceSize];

    input.close();
  } else {
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0)
    {
      return 1;
    }
  }

  calc(trace, traceSize, t0, dt, y0, y1, rank, size);

  if (rank == 0)
  {
    // Prepare output file
    std::ofstream output(argv[2]);
    if (!output.is_open())
    {
      std::cout << "[Error] Can't open " << argv[2] << " for read\n";
      delete trace;
      return 1;
    }

    for (uint32_t i = 0; i < traceSize; i++)
    {
      output << " " << trace[i];
    }
    output << std::endl;
    output.close();
    delete trace;
  }

  MPI_Finalize();
  return 0;
}
