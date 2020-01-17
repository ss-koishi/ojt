#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <random>
using namespace std;

vector< string > get_all_image(string path) {
  DIR *dp;
  dirent *entry;

  dp = opendir(path.c_str());
  vector< string > ret;
  do {
    entry = readdir(dp);
    if(entry != NULL) ret.push_back(entry->d_name);
  } while(entry != NULL);

  return ret;
}

int main(int argv, char *argv[]) {
  
  double train_p = atoi(argv[2]);
  double validation_p = atoi(argv[3]);
  
  if(train_p + validation_p > 0.8) {
    cout << "must be less than 0.8." << endl;
    exit(0);
  }

  vector< string > ok_list = get_all_image("./OK");
  vector< string > ng_list = get_all_image("./NG");

  int ok_num = ok_list.size();
  int ng_num = ng_list.size();

  int train_ok_num = (int)(ok_num * train_p);
  int validation_ok_num = (int)(ok_num * validation_p);i
  int test_ok_num = ok_num - (train_ok_num + validation_ok_num);

  random_device seed;
  mt19937 engine(sedd());
