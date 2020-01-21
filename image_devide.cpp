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

bool gray_mode = false;

int main(int argc, char *argv[]) {

  if(argc < 2) {
      cout << "error: number of arguments must be 1." << endl;
      exit(0);
  }

  if(argc == 4) {
      if(argv[3][0] == '1') gray_mode = true;
  }

  double train_p = atof(argv[1]);

  vector< string > ok_list = get_all_image("./OK");
  vector< string > ng_list = get_all_image("./NG");

  int ok_num = ok_list.size();
  int ng_num = ng_list.size();

  int train_ok_num = ok_num * train_p;
  int train_ng_num = ng_num * train_p;
  int validation_ok_num = ok_num - train_ok_num;
  int validation_ng_num = ng_num - train_ng_num;


  cout << "* class OK: " << ok_num << endl;
  cout << "--- train: " << train_ok_num << endl;
  cout << "--- validation: " << validation_ok_num << endl;
  cout << "* class NG: " << ng_num << endl;
  cout << "--- train: " << train_ng_num << endl;
  cout << "--- validation: " << validation_ng_num << endl;


  random_device seed;
  mt19937 engine(seed());
  shuffle(ok_list.begin(), ok_list.end(), engine);
  shuffle(ng_list.begin(), ng_list.end(), engine);

  cout << "rm -rf ./train ./validation" << endl;
  system("rm -rf ./train ./validation");
  cout << "mkdir -p train/OK train/NG" << endl;
  system("mkdir -p train/OK train/NG");
  cout << "mkdir -p validation/OK validation/NG" << endl;
  system("mkdir -p validation/OK validation/NG");

  for(int i = 0; i < ok_num; i++) {
      if(i < train_ok_num) {
          cout << "cp ./OK/" + ok_list[i] + " ./train/OK/" + ok_list[i] << endl;
          system(("cp ./OK/" + ok_list[i] + " ./train/OK/" + ok_list[i]).c_str());
          if(gray_mode) system(("convert ./train/OK/" + ok_list[i] + " -type GrayScale ./train/OK/" + ok_list[i]).c_str());
      } else {
          cout << "cp ./OK/" + ok_list[i] + " ./validation/OK/" + ok_list[i] << endl;
          system(("cp ./OK/" + ok_list[i] + " ./validation/OK/" + ok_list[i]).c_str());
          if(gray_mode) system(("convert ./validation/OK/" + ok_list[i] + " -type GrayScale ./validation/OK/" + ok_list[i]).c_str());
      }
  }

  cout << endl;
  for(int i = 0; i < ng_num; i++) {
      if(i < train_ng_num) {
          cout << "cp ./NG/" + ng_list[i] + " ./train/NG/" + ng_list[i] << endl;
          system(("cp ./NG/" + ng_list[i] + " ./train/NG/" + ng_list[i]).c_str());
          if(gray_mode) system(("convert ./train/NG/" + ng_list[i] + " -type GrayScale ./train/NG/" + ng_list[i]).c_str());
      } else {
          cout << "cp ./NG/" + ng_list[i] + " ./validation/NG/" + ng_list[i] << endl;
          system(("cp ./NG/" + ng_list[i] + " ./validation/NG/" + ng_list[i]).c_str());
          if(gray_mode) system(("convert ./validation/NG/" + ng_list[i] + " -type GrayScale ./validation/NG/" + ng_list[i]).c_str());
      }
  }
}
