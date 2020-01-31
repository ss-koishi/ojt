#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <random>
#include <cstring>
#include <unistd.h>
using namespace std;


bool gray_mode = false;
int class_num = 2;

string classes[] = {"OK", "NG", "STOP"};
vector< string > labels;
string base_path;

vector< string > get_all_image(string path) {
    cout << path << endl;
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

void argerror(char *argv[]) {
    cout << "Usage: " << argv[0] << "[-m argument] [-n argument] arg1 ..." << endl;
    cout << "-m: select a mode" << endl;
    cout << "  0: normal_image" << endl;
    cout << "  1: grayscale_image" << endl;
    cout << "-n: number of classes" << endl;
}

int parse_options(int argc, char *argv[]) {
    int opt, success = 1;
    opterr = 0;

    while((opt = getopt(argc, argv, "gn:")) != -1) {
        switch(opt) {
            case 'g':
                gray_mode = true;
                break;
            case 'n':
                if(strcmp(optarg, "2") == 0) class_num = 2;
                else if(strcmp(optarg, "3") == 0) class_num = 3;
                else success = 0;
                break;
            default:
                cout << "default" << endl;
                success = 0;
                break;
         }
    }

    return success;
}

void set_class(int n) {
    for(int i = 0; i < n; i++) {
        labels.push_back(classes[i]);
    }
    if(n == 2) base_path = "./images/class-2/";
    if(n == 3) base_path = "./images/class-3/";
    if(n == 4) base_path = "./images/class-4/";
}


int main(int argc, char *argv[]) {

  int success = parse_options(argc, argv);
  if(!success) exit(0);

  double train_p = atof(argv[argc - 1]);

  set_class(class_num);
  cout << "hoge " << endl;
  vector< vector< string > > image_list(class_num, vector< string > ());
  for(int i = 0; i < class_num; i++) {
      image_list[i] = get_all_image(base_path + labels[i] + "/");
  }
  cout << "hoge 1" << endl;
  vector< int > nums, train_nums, validation_nums;
  for(int i = 0; i < class_num; i++) {
      nums.push_back(image_list[i].size());
      train_nums.push_back((int)(nums[i] * train_p));
      validation_nums.push_back(nums[i] - train_nums[i]);
  }
  cout << "hoge 2" << endl;
  for(int i = 0; i < class_num; i++) {
      cout << "** class " + labels[i] + " :" << nums[i] << endl;
      cout << "---- train: " << train_nums[i] << endl;
      cout << "---- validation: " << validation_nums[i] << endl;
  }

  random_device seed;
  mt19937 engine(seed());
  for(int i = 0; i < class_num; i++) {
      shuffle(image_list[i].begin(), image_list[i].end(), engine);
  }

  cout << "rm -rf ./train ./validation" << endl;
  system("rm -rf ./train ./validation");
  for(int i = 0; i < class_num; i++) {
      string cmd = "mkdir -p train/" + labels[i];
      cout << cmd << endl;
      system(cmd.c_str());
      cmd = "mkdir -p validation/" + labels[i];
      cout << cmd << endl;
      system(cmd.c_str());
  }


  for(int i = 0; i < class_num; i++) {
      for(int j = 0; j < nums[i]; j++) {
          if(j < train_nums[i]) {
              string cmd = "cp " + base_path + labels[i] + "/" + image_list[i][j] + " ./train/" + labels[i] + "/" + image_list[i][j];
              cout << cmd << endl;
              system(cmd.c_str());
              if(gray_mode) system(("convert ./train/" + labels[i] + "/" + image_list[i][j] + " -type GrayScale ./train/" + labels[i] + "/" + image_list[i][j]).c_str());
          } else {
              string cmd = "cp " + base_path + labels[i] + "/" + image_list[i][j] + " ./validation/" + labels[i] + "/" + image_list[i][j];
              cout << cmd << endl;
              system(cmd.c_str());
              if(gray_mode) system(("convert ./validation/" + labels[i] + "/" + image_list[i][j] + " -type GrayScale ./validation/" + labels[i] + "/" + image_list[i][j]).c_str());
          }
      }
  }
}
