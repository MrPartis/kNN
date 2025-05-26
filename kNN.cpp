#include "kNN.hpp"

Dataset::Dataset(const Dataset& other){
    data = new dat<List<int>*>;
    for (int i = 0;i<other.data->length();i++){
        List<int>* tmp = new dat<int>();
        List<int>* cur = (other.data->get(i));
        for (int j = 0;j<cur->length();++j) tmp->push_back(cur->get(j));
        data->push_back(tmp);
        tmp = nullptr;
    }
    label = new dat<string>;
    for (int i = 0;i<other.label->length();++i) label->push_back(other.label->get(i));
}

Dataset& Dataset::operator=(const Dataset& other){
    int a = data->length();
    for (int i = 0;i<a;++i) {
        List<int>* x = data->get(i);
        x->clear();
        delete x;
    }
    data->clear();
    label->clear();
    for (int i = 0;i<other.data->length();++i){
        List<int>* b = new dat<int>;
        List<int>* c = other.data->get(i);
        for (int j = 0;j<c->length();++j) b->push_back(c->get(j));
        data->push_back(b);
        b = nullptr;
    }
    for (int i = 0;i<other.label->length();++i) label->push_back(other.label->get(i));
    return *this;
}

void Dataset::printHead(int nRows, int nCols) const{
    if (nRows>0 && nCols>0) {
        nRows = nRows>data->length()?data->length():nRows;
        nCols = nCols>label->length()?label->length():nCols;
        for (int i = 0;i<nCols;++i) cout << label->get(i) << (i!=nCols-1?" ":"");
        cout << '\n';
        for (int i = 0;i<nRows;++i) {
            List<int>* g = data->get(i);
            for (int j = 0;j<nCols;++j) cout << g->get(j) << (j!=nCols-1?" ":"");
            if (i!=nRows-1) cout << '\n';
        }
    }
}
void Dataset::printTail(int nRows, int nCols) const{
    if (nRows>0 && nCols>0) {
        nRows = nRows>data->length()?data->length():nRows;
        nCols = nCols>label->length()?label->length():nCols;
        for (int i = label->length()-nCols;i<label->length();++i) cout << label->get(i) << (i!=label->length()-1?" ":"");
        cout << '\n';
        for (int i = data->length()-1-nRows;i<data->length()-1;++i) {
            List<int>* g = data->get(i);
            for (int j = label->length()-nCols;j<label->length();++j) cout << g->get(j) << (j!=label->length()-1?" ":"");
            if (i!=data->length()-2) cout << '\n';
        }
    }
}
void Dataset::getShape(int& nRows, int& nCols) const{
    nRows = data->length();
    nCols = label->length();
}
void Dataset::columns() const{
    for (int i = 0;i<label->length();++i) cout << label->get(i) << (i!=label->length()-1?" ":"");
    cout << '\n';
}
bool Dataset::drop(int axis, int index, string columnName){
    if (axis>1 || axis<0) return false;
    if (axis==0){
        if (index>=data->length() || index<0) return false; else {
            List<int>* x = data->get(index);
            x->clear();
            data->remove(index);
            return true;
        }
    } else {
        if (index>=label->length()) return false; else {
            int in = 0;
            if (columnName!="label"){
                int pos = columnName.find('x');
                if (pos==-1) return false;
                string a = columnName.substr(0,pos);
                columnName.erase(0,pos+1);
                for (int i = 0;i<(int)a.length();++i) if (!(a[i]>='0' && a[i]<='9')) return false;
                for (int i = 0;i<(int)columnName.length();++i) if (!(columnName[i]>='0' && columnName[i]<='9')) return false;
                if (stoi(a)>0 && stoi(a)<=28 && stoi(columnName)>0 && stoi(columnName)<=28) in = (stoi(a)-1)*28+stoi(columnName); else return false;
            } else return false;
            for (int i = 0;i<data->length();++i){
                List<int>* x = data->get(i);
                x->remove(in+1);
            }
            label->remove(in+1);
            return true;
        }
    }
}
Dataset Dataset::extract(int startRow, int endRow, int startCol, int endCol) const{
    Dataset nw;
    List<List<int>*>* in = nw.getData();
    int takex = (endRow==-1)?data->length():endRow+1, takey = (endCol==-1)?label->length():endCol+1;
    for (int i = startRow;i<takex;++i) {
        dat<int>* a = new dat<int>;
        List<int>* b = data->get(i);
        for (int j = startCol;j<takey;++j) a->push_back(b->get(j));
        in->push_back(a);
        a = nullptr;
    }
    for (int i = startCol;i<takey;++i) nw.label->push_back(label->get(i));
    return nw;
}
List<List<int>*>* Dataset::getData() const{
    return data;
}

List<string>* Dataset::getLabel() const{
    return label;
}
bool Dataset::loadFromCSV(const char* fileName){
    ifstream in(fileName);
    if (!in.is_open()) return false;
    string inp,parts;
    while (!in.eof()){
        getline(in,inp,'\n');
        stringstream in2(inp);
        if (inp.substr(0,5)=="label") {
            while (getline(in2,parts,',')){
                label->push_back(parts);
                inp.erase(0,parts.length());
                if (inp.length()!=0 && inp[0]==',') inp.erase(0,1);
            }
        } else if (inp.length()!=0){
            dat<int>* x = new dat<int>();
            while (getline(in2,parts,',')) x->push_back(stoi(parts));
            data->push_back(x);
        }
    }
    return true;
}

kNN::kNN(int k){
    this->k = k;
    number_labelling = new dat<int>;
    data = new dat<List<int>*>;
    dist = new dat<int>;
    link = new dat<int>;
}
void kNN::fit(const Dataset& X_train, const Dataset& y_train){
    if (data->length()!=0){
        for (int i = 0;i<data->length();++i) {
            List<int>* x = data->get(i);
            x->clear();
        }
        data->clear();
        dist->clear();
    }
    List<List<int>*>* a = X_train.getData();
    List<List<int>*>* b = y_train.getData();
    for (int i = 0;i<a->length();++i){
         List<int>* c = a->get(i);
         dat<int>* c2 = new dat<int>;
         for (int j = 0;j<c->length();++j) c2->push_back(c->get(j));
         data->push_back(c2);
         c2 = nullptr;
         List<int>* d = b->get(i);
         number_labelling->push_back(d->get(0));
    }
}
Dataset kNN::predict(const Dataset& X_test){ //cần sửa
    Dataset answer; // answer for prediction
    answer.getLabel()->push_back("label");
    List<int>* con, *pend; // con: trained; pend: test
    List<List<int>*>* test = X_test.getData(), *ans = answer.getData();
    for (int i = 0;i<test->length();++i){
        dist->clear();
        List<int>* an = new dat<int>;
        pend = test->get(i);
        for (int j = 0;j<data->length();++j){
            con = data->get(j);
            double dis = 0,sub = 0;
            int z = 0; //counting
            while (z!=con->length()){ //calculate Euclidean distance
                int cmp = 0;
                if (dist->length()==this->k) cmp = dist->get(dist->length()-1);
                sub = 0;
                if (!(con->get(z)==0 && pend->get(z)==0)){
                    if (con->get(z)!=0) sub+=con->get(z);
                    if (pend->get(z)!=0) sub-=pend->get(z);
                    dis+=sub*sub;
                    if (dist->length()==this->k) if (dis>cmp){
                        z = con->length();
                        dis = 0;
                        break;
                    }
                }
                z++;
            }
            if (dis!=0) {
                if (dist->length()==0) { //insertion sort for finalized data
                    dist->push_back(dis);
                    link->push_back(number_labelling->get(j));
                } else {
                    z = 0;
                    int lim = dist->length()<k?dist->length():k;
                    while (z<lim){
                        if (dist->get(z)>dis){
                            dist->insert(z,dis);
                            link->insert(z,number_labelling->get(j));
                            z = lim;
                        } else z++;
                    }
                    if (dist->length()>this->k){
                        dist->remove(this->k);
                        link->remove(this->k);
                    } else {
                        dist->push_back(dis);
                        link->push_back(number_labelling->get(j));
                    }
                }
            }
        }
        int* x = new int[10]; //look at most accurate prediction
        for (int j = 0;j<this->k;++j) (*(x+link->get(j)))++;
        int mx = 0;
        for (int j = 1;j<10;++j) mx = (*(x+mx)<*(x+j))?j:mx;
        for (int j = 0;j<10;++j) *(x+j) = 0;
        an->push_back(mx); //insert data and remove traces
        ans->push_back(an);
        link->clear();
        an = nullptr;
    }
    return answer;
}
double kNN::score(const Dataset& y_test, const Dataset& y_pred){
    List<int>* x, *y;
    int correct = 0;
    for (int i = 0;i<y_test.getData()->length();++i){
        x = y_test.getData()->get(i);
        y = y_pred.getData()->get(i);
        if (x->get(0)==y->get(0)) correct++;
    }
    return ((double)correct)/y_test.getData()->length();
}

void train_test_split(Dataset& X, Dataset& y, double test_size, Dataset& X_train, Dataset& X_test, Dataset& y_train, Dataset& y_test){
    List<List<int>*>* in_fea = X.getData(); //base feature
    List<List<int>*>* in_lab = y.getData(); //base label
    List<List<int>*>* a = X_train.getData();//train feature
    List<List<int>*>* b = X_test.getData(); //test feature
    List<List<int>*>* c = y_train.getData();//train label
    List<List<int>*>* d = y_test.getData(); //test label
    int take = (int)ceil(test_size*(in_fea->length()));
    for (int i = 0;i<in_fea->length();++i) {
        List<int>* e = in_fea->get(i); //inner base feature
        List<int>* f = in_lab->get(i); //inner base label
        dat<int>* g = new dat<int>;
        dat<int>* h = new dat<int>;
        List<string>* k = (i==in_fea->length()-take)?y_train.getLabel():y_test.getLabel();
        List<string>* l = X.getLabel();
        if (i<in_fea->length()-take){
            if (i==0) k->push_back("label");
            for (int j = 0;j<e->length();++j) {
                g->push_back(e->get(j));
                if (j==0) h->push_back(f->get(0));
                if (i==0) X_train.getLabel()->push_back(l->get(j));
            }
            a->push_back(g);
            c->push_back(h);
        } else {
            if (i==in_fea->length()-take) k->push_back("label");
            for (int j = 0;j<e->length();++j) {
                g->push_back(e->get(j));
                if (j==0) h->push_back(f->get(0));
                if (i==in_fea->length()-take) X_test.getLabel()->push_back(l->get(j));
            }
            b->push_back(g);
            d->push_back(h);
        }
        g = nullptr;
        h = nullptr;
    }
}
