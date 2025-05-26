#include "main.hpp"

/* TODO: Please design your data structure carefully so that you can work with the given dataset
 *       in this assignment. The below structures are just some suggestions.
 */
template<typename T>
class List {
public:
    class node{
    private:
        T data;
    public:
        node* next;
        node* prev;
        void set(T dat){
            data = dat;
        }
        T get(){
            return data;
        }
        T& get_ref(){
            return data;
        }
        node(T value){
            data = value;
            next = nullptr;
            prev = nullptr;
        };
        node(T value, node* nx, node* pr){
            data = value;
            next = nx;
            prev = pr;

        }
    };
    node* head;
    node* tail;
    int count;
    virtual ~List() = default;
    virtual void push_back(T value) = 0;
    virtual void push_front(T value) = 0;
    virtual void insert(int index, T value) = 0;
    virtual void remove(int index) = 0;
    virtual T& get(int index) const = 0;
    virtual int length() const = 0 ;
    virtual void clear() = 0;
    virtual void print() const = 0;
    virtual void reverse() = 0;
};

template <typename T>
class dat : public List<T> {
public:
    dat(){
        this->head = nullptr;
        this->tail = nullptr;
        this->count = 0;
    }
    ~dat() = default;
    void push_back(T value) {
        typename List<T>::node* g = new typename List<T>::node(value);
        if (this->count==0){
            delete this->head;
            this->head = g;
            this->tail = g;
        } else {
            this->tail->next = g;
            g->prev = this->tail;
            this->tail = g;
        }
        (this->count)++;
    }
    void push_front(T value){
        typename List<T>::node* g = new typename List<T>::node(value);
        if (this->count==0){
            this->head = g;
            this->tail = g;
        } else {
            this->head->prev = g;
            g->next = this->head;
            this->head = g;
        }
        this->count++;
    }
    void insert(int index, T value){
        typename List<T>::node* g = new typename List<T>::node(value);
        if (index>=0 && index<=this->count){
            if (index==0) {
                if (this->head!=nullptr){
                    this->head->prev = g;
                    g->next = this->head;
                } else this->tail = g;
                this->head = g;
            } else {
                typename List<T>::node* x = this->head;
                for (int i = 0;i<index-1;++i) x = x->next;
                if (x->next!=nullptr) {
                    x->next->prev = g;
                    g->next = x->next;
                } else {
                    this->tail = g;
                }
                g->prev = x;
                x->next = g;
            }
        }
        this->count++;
    }
    void remove(int index){
        if (index<0 || index>=this->count) return;
        typename List<T>::node* x;
        if (this->count>1){
            if (index == 0){
                x = this->head;
                this->head = x->next;
                this->head->prev = nullptr;
            } else if (index==this->count-1) {
                x = this->tail;
                this->tail = x->prev;
                this->tail->next = nullptr;
            } else {
                x = this->head;
                for (int i = 0;i<index;++i) x = x->next;
                x->prev->next = x->next;
                x->next->prev = x->prev;
            }
        } else {
            x = this->head;
            this->head = nullptr;
            this->tail = nullptr;
        }
        delete x;
        this->count--;
    }
    T& get(int index) const{
        if (index<0 || index>=this->count) {cout << "Error at " << index << endl; throw std::out_of_range("get(): Out of range");}
        typename List<T>::node* x = this->head;
        for (int i = 0;i<index;++i) x = x->next;
        return x->get_ref();
    }
    int length() const{
        return this->count;
    }
    void clear(){
        if (this->count==0) return;
        typename List<T>::node* trace = this->head;
        while (trace!=this->tail){
            trace = trace->next;
            delete trace->prev;
        }
        if (this->tail!=nullptr) delete this->tail;
        this->head = nullptr;
        this->tail = nullptr;
        this->count = 0;
    }
    void print() const{
        typename List<T>::node* a = this->head;
        while (a!=nullptr) {
            cout << a->get() << (a->next==nullptr?"":" ");
            a = a->next;
        }
        cout << endl;
    }
    void reverse(){
        if (this->head==this->tail) return;
        typename List<T>::node* a = this->head;
        typename List<T>::node* b = this->tail;
        typename List<T>::node* c;
        while (!(a==b || b->next == a)){
            c = new typename List<T>::node(a->get());
            a->set(b->get());
            b->set(c->get());
            delete c;
            a = a->next;
            b = b->prev;
        }
    }
};

class Dataset {
private:
    List<List<int>*>* data;
    List<string>* label; //column labelling
    //You may need to define more
public:
    Dataset(){
        label = new dat<string>;
        data = new dat<List<int>*>;
    }
    ~Dataset() = default;
    Dataset(const Dataset& other);
    Dataset& operator=(const Dataset& other);
    bool loadFromCSV(const char* fileName);
    void printHead(int nRows = 5, int nCols = 5) const;
    void printTail(int nRows = 5, int nCols = 5) const;
    void getShape(int& nRows, int& nCols) const;
    void columns() const;
    bool drop(int axis = 0, int index = 0, std::string columns = "");
    Dataset extract(int startRow = 0, int endRow = -1, int startCol = 0, int endCol = -1) const;
    List<List<int>*>* getData() const;
    List<string>* getLabel() const;
};

class kNN {
private:
    int k;
    List<int>* number_labelling;
    List<List<int>*>* data;
    List<int>* dist;
    List<int>* link;
public:
    kNN(int k = 5);
    void fit(const Dataset& X_train, const Dataset& y_train);
    Dataset predict(const Dataset& X_test);
    double score(const Dataset& y_test, const Dataset& y_pred);
};

void train_test_split(Dataset& X, Dataset& y, double test_size, Dataset& X_train, Dataset& X_test, Dataset& y_train, Dataset& y_test);

// Please add more or modify as needed
