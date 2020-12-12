class Student:
  def __init__(self, name, listOfMarks):
    self.name = name
    self.listOfMarks = listOfMarks


def retrieve_students():
    f = open("students.txt", "r")
    listOfStudents = []
    for student in f:
        studentData = student.split()
        listOfMarksForStudent = []
        name = studentData[0]
        for x in range (1,len(studentData)):
            listOfMarksForStudent.append(int(studentData[x]))
        newStudent = Student(name, listOfMarksForStudent)
        listOfStudents.append(newStudent)
    return listOfStudents
