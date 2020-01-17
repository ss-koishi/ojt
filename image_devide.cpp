#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <random>
#include <cstring>
using namespace std;

vector< string > get_all_image(string path) {
  DIR *dp;
  dirent *entry;

  dp = opendir(path.c_str());
  vector< string > ret;
  do {
    entry = readdir(dp);
    if(entry != NULL && strlen(entry->d_name) > 4) ret.push_back(entry->d_name);
  } while(entry != NULL);

  return ret;
}

int main(int argc, char *argv[]) {

  if(argc < 3) {
      cout << "error: number of arguments must be 2." << endl;
      exit(0);
  }

  double train_p = atof(argv[1]);
  double validation_p = atof(argv[2]);

  if(train_p + validation_p > 0.9) {
    cout << "must be less than 0.9." << endl;
    exit(0);
  }

  vector< string > ok_list = get_all_image("./OK");
  vector< string > ng_list = get_all_image("./NG");

  int ok_num = ok_list.size();
  int ng_num = ng_list.size();

  int train_ok_num = ok_num * train_p;
  int validation_ok_num = ok_num * validation_p;
  int test_ok_num = ok_num - (train_ok_num + validation_ok_num);

  int train_ng_num = ng_num * train_p;
  int validation_ng_num = ng_num * validation_p;
  int test_ng_num = ng_num - (train_ng_num + validation_ng_num);


  cout << "* class OK: " << ok_num << endl;
  cout << "--- train: " << train_ok_num << endl;
  cout << "--- validation: " << validation_ok_num << endl;
  cout << "--- test: " << test_ok_num << endl;
  cout << "* class NG: " << ng_num << endl;
  cout << "--- train: " << train_ng_num << endl;
  cout << "--- validation: " << validation_ng_num << endl;
  cout << "--- test: " << test_ng_num << endl;

  random_device seed;
  mt19937 engine(seed());
  shuffle(ok_list.begin(), ok_list.end(), engine);
  shuffle(ng_list.begin(), ng_list.end(), engine);

  cout << "rm -rf ./train ./validation ./test" << endl;
  system("rm -rf ./train ./validation ./test");
  cout << "mkdir -p train/OK train/NG" << endl;
  system("mkdir -p train/OK train/NG");
  cout << "mkdir -p validation/OK validation/NG" << endl;
  system("mkdir -p validation/OK validation/NG");
  cout << "mkdir -p test/OK test/NG" << endl;
  system("mkdir -p test/OK test/NG");

  for(int i = 0; i < ok_num; i++) {
      if(i < train_ok_num) {
          cout << "cp ./OK/" + ok_list[i] + " ./train/OK/" + ok_list[i] << endl;
          system(("cp ./OK/" + ok_list[i] + " ./train/OK/" + ok_list[i]).c_str());
      } else if(i < train_ok_num + validation_ok_num) {
          cout << "cp ./OK/" + ok_list[i] + " ./validation/OK/" + ok_list[i] << endl;
          system(("cp ./OK/" + ok_list[i] + " ./validation/OK/" + ok_list[i]).c_str());
      } else {
          cout << "cp ./OK/" + ok_list[i] + " ./test/OK/" + ok_list[i] << endl;
          system(("cp ./OK/" + ok_list[i] + " ./test/OK/" + ok_list[i]).c_str());
      }
  }

  cout << endl;
  for(int i = 0; i < ng_num; i++) {
      if(i < train_ng_num) {
          cout << "cp ./NG/" + ng_list[i] + " ./train/NG/" + ng_list[i] << endl;
          system(("cp ./NG/" + ng_list[i] + " ./train/NG/" + ng_list[i]).c_str());
      } else if(i < train_ng_num + validation_ng_num) {
          cout << "cp ./NG/" + ng_list[i] + " ./validation/NG/" + ng_list[i] << endl;
          system(("cp ./NG/" + ng_list[i] + " ./validation/NG/" + ng_list[i]).c_str());
      } else {
          cout << "cp ./NG/" + ng_list[i] + " ./test/NG/" + ng_list[i] << endl;
          system(("cp ./NG/" + ng_list[i] + " ./test/NG/" + ng_list[i]).c_str());
      }
  }
}
